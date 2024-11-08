import logging
import threading
import time
from typing import Any, Dict, List, Tuple, Optional, NamedTuple

import cflib.crtp
import numpy as np
import torch
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from phoenix_drone_simulation.utils.utils import load_actor_critic_and_env_from_disk


class DroneState(NamedTuple):
    """Container for drone state variables"""
    position: np.ndarray
    quaternions: np.ndarray
    linear_vel: np.ndarray
    angular_vel: np.ndarray
    position_error: np.ndarray
    last_actions: np.ndarray


class ScalingParams:
    """Container for scaling parameters"""
    # Position scaling (real: [-1.5, 1.5]m -> sim: [-1.0, 1.0])
    POS_SCALE = 1.5
    # Altitude scaling (real: [0.8, 1.2]m -> sim: [0.0, 1.0])
    ALT_MIN, ALT_MAX = 0.8, 1.2
    # Angular velocity scaling (real: [-250, 250]°/s -> sim: [-1.0, 1.0])
    ANG_VEL_SCALE = 250.0
    # Action scaling (from simulation to real)
    THRUST_MIN, THRUST_MAX = 35000, 55000  # Real thrust range
    ATTITUDE_LIMIT = 10.0  # Real attitude limit in degrees
    YAW_RATE_LIMIT = 10.0  # Real yaw rate limit in deg/s


def setup_logging() -> logging.Logger:
    """Configure logging settings"""
    logging.basicConfig(
        filename='crazyflie.log',
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def create_log_configs(cf: Crazyflie) -> List[LogConfig]:
    """Create multiple logging configurations to stay within size limits"""
    log_configs = []

    # Position + Quaternions
    pos_quat_config = LogConfig(name="pos_quat", period_in_ms=10)
    pos_quat_config.add_variable("kalman.stateX", "float")
    pos_quat_config.add_variable("kalman.stateY", "float")
    pos_quat_config.add_variable("kalman.stateZ", "float")
    pos_quat_config.add_variable("kalman.q0", "float")
    pos_quat_config.add_variable("kalman.q1", "float")
    pos_quat_config.add_variable("kalman.q2", "float")
    pos_quat_config.add_variable("kalman.q3", "float")
    log_configs.append(pos_quat_config)

    # Linear velocities
    vel_config = LogConfig(name="velocities", period_in_ms=10)
    vel_config.add_variable("kalman.statePX", "float")
    vel_config.add_variable("kalman.statePY", "float")
    vel_config.add_variable("kalman.statePZ", "float")
    log_configs.append(vel_config)

    # Angular velocities
    gyro_config = LogConfig(name="gyro", period_in_ms=10)
    gyro_config.add_variable("gyro.x", "float")
    gyro_config.add_variable("gyro.y", "float")
    gyro_config.add_variable("gyro.z", "float")
    log_configs.append(gyro_config)

    return log_configs


def scale_action_to_real(action: np.ndarray) -> Tuple[float, float, float, float]:
    """Scale neural network outputs to real drone commands"""
    # Thrust scaling (from [-1, 1] to [THRUST_MIN, THRUST_MAX])
    thrust = (action[0] + 1.0) * 0.5 * (ScalingParams.THRUST_MAX - ScalingParams.THRUST_MIN) + ScalingParams.THRUST_MIN

    # Attitude scaling
    roll = action[1] * ScalingParams.ATTITUDE_LIMIT
    pitch = action[2] * ScalingParams.ATTITUDE_LIMIT
    yaw_rate = action[3] * ScalingParams.YAW_RATE_LIMIT

    return (
        float(np.clip(thrust, ScalingParams.THRUST_MIN, ScalingParams.THRUST_MAX)),
        float(np.clip(roll, -ScalingParams.ATTITUDE_LIMIT, ScalingParams.ATTITUDE_LIMIT)),
        float(np.clip(pitch, -ScalingParams.ATTITUDE_LIMIT, ScalingParams.ATTITUDE_LIMIT)),
        float(np.clip(yaw_rate, -ScalingParams.YAW_RATE_LIMIT, ScalingParams.YAW_RATE_LIMIT))
    )


class ObservationHandler:
    """Handles observation processing, scaling and standardization"""

    def __init__(self, model_path: str):
        # Load standardization parameters (mean, std) from model
        try:
            scaling_data = np.load(f"{model_path}/observation_scaling.npz")
            self.obs_mean = scaling_data['mean']
            self.obs_std = scaling_data['std']
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Could not load scaling parameters: {e}. Using defaults.")
            self.obs_mean = np.zeros(40)  # Full observation space
            self.obs_std = np.ones(40)

        self.latest_data = {
            'pos_quat': None,
            'velocities': None,
            'gyro': None,
            'timestamp': None
        }
        self.lock = threading.Lock()
        self.previous_obs = None

    def update_data(self, group: str, timestamp: float, data: Dict[str, float]):
        """Update latest sensor data for a group"""
        with self.lock:
            self.latest_data[group] = data
            self.latest_data['timestamp'] = timestamp

    def construct_observation(self, last_actions: np.ndarray) -> Optional[np.ndarray]:
        """Construct scaled and standardized observation"""
        with self.lock:
            # Check if we have all required data
            if any(v is None for v in self.latest_data.values()):
                return None

            # Position (scale to [-1, 1])
            position = np.array([
                self.latest_data['pos_quat']['kalman.stateX'],
                self.latest_data['pos_quat']['kalman.stateY'],
                self.latest_data['pos_quat']['kalman.stateZ']
            ])
            position[:2] = np.clip(position[:2] / ScalingParams.POS_SCALE, -1.0, 1.0)
            position[2] = (position[2] - ScalingParams.ALT_MIN) / (ScalingParams.ALT_MAX - ScalingParams.ALT_MIN)

            # Quaternions (already in [-1, 1])
            quaternions = np.array([
                self.latest_data['pos_quat']['kalman.q0'],
                self.latest_data['pos_quat']['kalman.q1'],
                self.latest_data['pos_quat']['kalman.q2'],
                self.latest_data['pos_quat']['kalman.q3']
            ])

            # Linear velocities (assumed to match sim scale)
            linear_vel = np.array([
                self.latest_data['velocities']['kalman.statePX'],
                self.latest_data['velocities']['kalman.statePY'],
                self.latest_data['velocities']['kalman.statePZ']
            ])

            # Angular velocities (scale to [-1, 1])
            angular_vel = np.array([
                self.latest_data['gyro']['gyro.x'],
                self.latest_data['gyro']['gyro.y'],
                self.latest_data['gyro']['gyro.z']
            ])
            angular_vel = np.clip(angular_vel / ScalingParams.ANG_VEL_SCALE, -1.0, 1.0)

            # Calculate position error
            timestamp = self.latest_data['timestamp']
            alpha = 2.0 * np.pi / 3.0
            target = np.array([
                0.25 * (1.0 - np.cos(timestamp * alpha)),
                0.25 * np.sin(timestamp * alpha),
                1.0
            ])
            position_error = target - position[:3]  # Use unscaled position for error

            # Combine all components for current timestep
            current_obs = np.concatenate([
                position,
                quaternions,
                linear_vel,
                angular_vel,
                position_error,
                last_actions
            ])

            # Handle previous observation
            if self.previous_obs is None:
                full_obs = np.concatenate([current_obs, current_obs])
            else:
                full_obs = np.concatenate([self.previous_obs, current_obs])

            # Store current observation for next timestep
            self.previous_obs = current_obs

            # Standardize observation
            obs_standardized = (full_obs - self.obs_mean) / self.obs_std

            return obs_standardized


class CrazyflieController:
    def __init__(self, uri: str = 'radio://0/80/2M', model_path: str = None):
        self.uri = uri
        self.logger = logger
        self.cf = Crazyflie()
        self.is_connected = True
        self.last_model_outputs = np.zeros(4, dtype=np.float32)

        # Create observation handler
        self.obs_handler = ObservationHandler(model_path)
        self.model_path = model_path

        # Setup callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup connection callbacks"""
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

    def _connected(self, uri: str):
        self.logger.info(f"Connected to {uri}")
        self._start_logging()
        threading.Thread(target=self._control_loop, daemon=True).start()

    def _disconnected(self, uri: str):
        self.logger.info(f"Disconnected from {uri}")
        self.is_connected = False

    def _connection_failed(self, uri: str, msg: str):
        self.logger.error(f"Connection to {uri} failed: {msg}")
        self.is_connected = False

    def _connection_lost(self, uri: str, msg: str):
        self.logger.warning(f"Connection to {uri} lost: {msg}")
        self.is_connected = False

    def _log_data_callback(self, timestamp: float, data: Dict[str, Any], logconf: LogConfig):
        """Process incoming sensor data"""
        self.obs_handler.update_data(logconf.name, timestamp, data)

    def _start_logging(self):
        """Start all logging configurations"""
        log_configs = create_log_configs(self.cf)

        for config in log_configs:
            try:
                self.cf.log.add_config(config)
                config.data_received_cb.add_callback(self._log_data_callback)
                config.start()
                self.logger.info(f"Started logging config: {config.name}")
            except AttributeError as e:
                self.logger.error(f"Could not add log config {config.name}: {e}")
            except KeyError as e:
                self.logger.error(f"Could not start log config {config.name}: {e}")

            time.sleep(0.1)

    def _control_loop(self):
        """Main control loop"""
        self.cf.commander.send_setpoint(0, 0, 0, 0)  # Unlock the safety lock
        time.sleep(0.1)

        start_time = time.time()
        takeoff_duration = 3.0
        interval = 0.01

        while self.is_connected:
            loop_start = time.time()
            elapsed = time.time() - start_time

            if elapsed < takeoff_duration:
                # Takeoff phase
                commands = (46000, 0.0, 0.0, 0.0)
                self.last_model_outputs = np.zeros(4)
                self.logger.debug(f"Takeoff phase: {elapsed:.2f}s, thrust={commands[0]}")
            else:
                # Neural network control phase
                obs = self.obs_handler.construct_observation(self.last_model_outputs)
                self.logger.debug(f"Observation: {obs}")

                if obs is not None:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        action, *_ = ac(obs_tensor)
                        self.last_model_outputs = action.numpy()
                        commands = scale_action_to_real(action)
                else:
                    commands = (35000, 0.0, 0.0, 0.0)  # Safe default
                    self.last_model_outputs = np.zeros(4)
                    self.logger.warning("No observation available")

            thrust, roll, pitch, yaw_rate = commands
            self.cf.commander.send_setpoint(roll, pitch, yaw_rate, int(thrust))
            self.logger.debug(
                f"Sending setpoint: roll={roll:.2f}, pitch={pitch:.2f}, yaw_rate={yaw_rate:.2f}, thrust={thrust:.0f}")

            # Maintain timing
            elapsed_loop = time.time() - loop_start
            if elapsed_loop < interval:
                time.sleep(interval - elapsed_loop)
            else:
                self.logger.warning(f"Loop took longer than interval: {elapsed_loop:.4f}s")

        self.cf.commander.send_stop_setpoint()
        self.logger.info("Control loop stopped")

    def start(self):
        """Start the Crazyflie controller"""
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(self.uri)

    def stop(self):
        """Stop the Crazyflie controller"""
        self.is_connected = False
        self.cf.close_link()


def main():
    global ac
    # Load the trained model
    model_path = "C:/Users/the_3/DroneRL/modules/phoenix-pybullet/saves/DroneCircleBulletEnv-v0/ppo/AttitudeMod/seed_65025"
    ac, _ = load_actor_critic_and_env_from_disk(model_path)

    # Create and start controller
    controller = CrazyflieController(model_path=model_path)
    try:
        controller.start()
        while controller.is_connected:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.stop()
        print("Interrupted by user. Shutting down.")


if __name__ == '__main__':
    main()
