import logging
import os
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
    THRUST_MIN, THRUST_MAX = 40000, 60000  # Real thrust range
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
    """Create multiple smaller logging configurations"""
    log_configs = []

    # Position only
    pos_config = LogConfig(name="position", period_in_ms=10)
    pos_config.add_variable("kalman.stateX", "float")
    pos_config.add_variable("kalman.stateY", "float")
    pos_config.add_variable("kalman.stateZ", "float")
    log_configs.append(pos_config)

    # Quaternions only
    quat_config = LogConfig(name="quaternion", period_in_ms=10)
    quat_config.add_variable("kalman.q0", "float")
    quat_config.add_variable("kalman.q1", "float")
    quat_config.add_variable("kalman.q2", "float")
    quat_config.add_variable("kalman.q3", "float")
    log_configs.append(quat_config)

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


def real_to_scaled_position(real_pos: np.ndarray) -> np.ndarray:
    """Convert real-world position to scaled position for model input"""
    scaled_pos = np.array(real_pos)  # Copy to avoid modifying input
    scaled_pos[0:2] = np.clip(scaled_pos[0:2] / ScalingParams.POS_SCALE, -1.0, 1.0)  # Scale and clip x,y
    scaled_pos[2] = (scaled_pos[2] - ScalingParams.ALT_MIN) / (ScalingParams.ALT_MAX - ScalingParams.ALT_MIN)  # Scale z
    return scaled_pos


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
        scaling_path = f"{model_path}/observation_scaling.npz" if model_path else None
        if scaling_path and os.path.exists(scaling_path):
            scaling_data = np.load(scaling_path)
            self.obs_mean = scaling_data['mean']
            self.obs_std = scaling_data['std']
        else:
            logger.warning("Using default scaling parameters")
            self.obs_mean = np.zeros(40)
            self.obs_std = np.ones(40)

        self.latest_data = {
            'position': None,
            'quaternion': None,
            'velocities': None,
            'gyro': None,
            'timestamp': None
        }
        self.lock = threading.Lock()
        self.previous_obs = None
        self.target_position = np.array([0.0, 0.0, 1.0])  # Default target

    def real_to_scaled_position(self, real_pos: np.ndarray) -> np.ndarray:
        """Convert real-world position to scaled position for model input"""
        scaled_pos = np.array(real_pos)  # Copy to avoid modifying input
        scaled_pos[0:2] = np.clip(scaled_pos[0:2] / ScalingParams.POS_SCALE, -1.0, 1.0)  # Scale and clip x,y
        scaled_pos[2] = (scaled_pos[2] - ScalingParams.ALT_MIN) / (
                ScalingParams.ALT_MAX - ScalingParams.ALT_MIN)  # Scale z
        return scaled_pos

    def update_data(self, group: str, timestamp: float, data: Dict[str, float]):
        """Update latest sensor data for a group"""
        with self.lock:
            self.latest_data[group] = data
            self.latest_data['timestamp'] = timestamp
            logger.debug(f"Updated {group} data: {data}")

    def construct_observation(self, last_actions: np.ndarray) -> Optional[np.ndarray]:
        """Construct scaled and standardized observation"""
        with self.lock:
            # Check if we have all required data
            if any(v is None for v in self.latest_data.values()):
                missing = [k for k, v in self.latest_data.items() if v is None]
                logger.warning(f"Missing data from: {missing}")
                return None

            try:
                # Get current position and scale it
                position = np.array([
                    self.latest_data['position']['kalman.stateX'],
                    self.latest_data['position']['kalman.stateY'],
                    self.latest_data['position']['kalman.stateZ']
                ])
                scaled_position = self.real_to_scaled_position(position)

                # Scale target position
                scaled_target = self.real_to_scaled_position(self.target_position)

                # Calculate position error in scaled space
                position_error = scaled_target - scaled_position

                # Quaternions (already in [-1, 1])
                quaternions = np.array([
                    self.latest_data['quaternion']['kalman.q0'],
                    self.latest_data['quaternion']['kalman.q1'],
                    self.latest_data['quaternion']['kalman.q2'],
                    self.latest_data['quaternion']['kalman.q3']
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

                # Combine all components for current timestep
                current_obs = np.concatenate([
                    scaled_position,
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

                logger.debug(f"Successfully constructed observation: {obs_standardized}")
                return obs_standardized

            except Exception as e:
                logger.error(f"Error constructing observation: {e}")
                return None


class CrazyflieController:
    def __init__(self, uri: str = 'radio://0/80/2M', model_path: str = None):
        self.uri = uri
        self.logger = logger
        self.cf = Crazyflie()
        self.is_connected = True
        self.last_model_outputs = np.zeros(4, dtype=np.float32)

        # Add parameters for back and forth motion
        self.start_position = np.array([0.0, 0.0, 1.0])  # Starting position
        self.motion_amplitude = 0.3  # 30cm back and forth
        self.motion_period = 4.0  # seconds for one complete cycle
        self.lookahead_time = 0.5  # seconds to look ahead for target
        self.start_time = None

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
        try:
            self.obs_handler.update_data(logconf.name, timestamp, data)
            self.logger.debug(f"Received {logconf.name} data: {data}")
        except Exception as e:
            self.logger.error(f"Error processing {logconf.name} data: {e}")

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

    def get_target_position(self, current_time: float) -> np.ndarray:
        """Calculate target position based on time"""
        if self.start_time is None:
            self.start_time = current_time

        elapsed = current_time - self.start_time
        phase = 2.0 * np.pi * elapsed / self.motion_period

        # Calculate current position in cycle
        x_offset = self.motion_amplitude * np.sin(phase)

        # Add lookahead to help model track better
        phase_ahead = 2.0 * np.pi * (elapsed + self.lookahead_time) / self.motion_period
        x_target = self.motion_amplitude * np.sin(phase_ahead)

        return np.array([
            self.start_position[0] + x_target,  # Move in x direction
            self.start_position[1],  # Fixed y
            self.start_position[2]  # Fixed height
        ])

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

            # Log all current sensor data
            with self.obs_handler.lock:
                self.logger.debug(f"Current sensor data: {self.obs_handler.latest_data}")

            if elapsed < takeoff_duration:
                # Takeoff phase
                commands = (46000, 0.0, 0.0, 0.0)
                self.last_model_outputs = np.zeros(4)
                self.logger.debug(f"Takeoff phase: {elapsed:.2f}s, thrust={commands[0]}")
            else:
                # Update target based on current time
                current_target = self.get_target_position(loop_start)
                self.obs_handler.target_position = current_target

                # Neural network control phase
                obs = self.obs_handler.construct_observation(self.last_model_outputs)

                if obs is not None:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        action, *_ = ac(obs_tensor)
                        self.last_model_outputs = action
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
        self.logger.info("Closing link")


def main():
    global ac
    # Load the trained model
    model_path = "C:/Users/the_3/DroneRL/modules/phoenix-pybullet/saves/DroneCircleBulletEnv-v0/ppo/AttitudeMod/seed_65025"
    try:
        ac, _ = load_actor_critic_and_env_from_disk(model_path)
        logger.info("Successfully loaded model from disk")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Create and start controller
    controller = CrazyflieController(model_path=model_path)
    try:
        controller.start()
        while controller.is_connected:
            time.sleep(1)
            # Print periodic status updates
            logger.info("Controller running - press Ctrl+C to stop")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        controller.stop()
        logger.info("Shutting down")


if __name__ == '__main__':
    main()
