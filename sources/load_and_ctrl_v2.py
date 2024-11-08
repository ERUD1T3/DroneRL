import logging
import threading
import time
from typing import Any, Dict, Tuple, NamedTuple

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


def scale_position(pos: np.ndarray) -> np.ndarray:
    """Scale position from real to simulation space"""
    # XY scaling
    pos[:2] = np.clip(pos[:2] / ScalingParams.POS_SCALE, -1.0, 1.0)
    # Z scaling
    pos[2] = (pos[2] - ScalingParams.ALT_MIN) / (ScalingParams.ALT_MAX - ScalingParams.ALT_MIN)
    return pos


def scale_angular_velocity(ang_vel: np.ndarray) -> np.ndarray:
    """Scale angular velocity from real to simulation space"""
    return np.clip(ang_vel / ScalingParams.ANG_VEL_SCALE, -1.0, 1.0)


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


def construct_observation(data: Dict[str, float], timestamp: float, last_actions: np.ndarray) -> np.ndarray:
    """Construct the scaled observation vector from sensor data"""
    # Extract position and scale
    position = np.array([
        data.get('kalman.stateX', 0.0),
        data.get('kalman.stateY', 0.0),
        data.get('kalman.stateZ', 0.0)
    ])
    position = scale_position(position)

    # Extract quaternions (already in correct range [-1, 1])
    quaternions = np.array([
        data.get('kalman.q0', 0.0),
        data.get('kalman.q1', 0.0),
        data.get('kalman.q2', 0.0),
        data.get('kalman.q3', 0.0)
    ])

    # Extract and scale linear velocities
    linear_vel = np.array([
        data.get('kalman.statePX', 0.0),
        data.get('kalman.statePY', 0.0),
        data.get('kalman.statePZ', 0.0)
    ])
    # Linear velocities are assumed to match simulation scale

    # Extract and scale angular velocities
    angular_vel = np.array([
        data.get('gyro.x', 0.0),
        data.get('gyro.y', 0.0),
        data.get('gyro.z', 0.0)
    ])
    angular_vel = scale_angular_velocity(angular_vel)

    # Calculate scaled position error
    alpha = 2.0 * np.pi / 3.0
    target = np.array([
        0.25 * (1.0 - np.cos(timestamp * alpha)),
        0.25 * np.sin(timestamp * alpha),
        1.0
    ])
    position_error = target - position

    # Combine all components
    return np.concatenate([
        position,
        quaternions,
        linear_vel,
        angular_vel,
        position_error,
        last_actions
    ])


class CrazyflieController:
    def __init__(self, uri: str = 'radio://0/80/2M'):
        self.uri = uri
        self.logger = setup_logging()
        self.cf = Crazyflie()
        self.is_connected = True
        self.latest_obs = None
        self.last_model_outputs = np.zeros(4, dtype=np.float32)
        self.obs_lock = threading.Lock()
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup Crazyflie callback functions"""
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
        with self.obs_lock:
            static_obs = self.latest_obs.copy() if self.latest_obs is not None else None
            current_obs = construct_observation(data, timestamp, self.last_model_outputs)

            if static_obs is not None:
                self.latest_obs = np.concatenate([static_obs[:20], current_obs])
            else:
                self.latest_obs = np.concatenate([current_obs, current_obs])

            self.logger.debug(f'Latest observation: {self.latest_obs}, shape={self.latest_obs.shape}')

    def _start_logging(self):
        """Configure and start sensor logging"""
        variables = [
            ('kalman.stateX', 'float'), ('kalman.stateY', 'float'),
            ('kalman.stateZ', 'float'), ('kalman.q0', 'float'),
            ('kalman.q1', 'float'), ('kalman.q2', 'float'),
            ('kalman.q3', 'float'), ('kalman.statePX', 'float'),
            ('kalman.statePY', 'float'), ('kalman.statePZ', 'float'),
            ('gyro.x', 'float'), ('gyro.y', 'float'), ('gyro.z', 'float')
        ]

        for idx, var_chunk in enumerate([variables[i:i + 7] for i in range(0, len(variables), 7)]):
            log_conf = LogConfig(name=f'StateEstimation_{idx}', period_in_ms=10)

            for var_name, var_type in var_chunk:
                log_conf.add_variable(var_name, var_type)

            self.cf.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(self._log_data_callback)
            log_conf.start()
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
                commands = (40000, 0.0, 0.0, 0.0)
                self.last_model_outputs = np.zeros(4)
                self.logger.debug(f'Takeoff phase: {elapsed:.2f}s, thrust={commands[0]}')
            else:
                # Neural network control phase
                with self.obs_lock:
                    if self.latest_obs is not None:
                        obs_tensor = torch.as_tensor(self.latest_obs, dtype=torch.float32)
                        with torch.no_grad():
                            action, *_ = ac(obs_tensor)
                            self.last_model_outputs = action.numpy()
                            commands = scale_action_to_real(action)
                    else:
                        commands = (10000, 0.0, 0.0, 0.0)
                        self.last_model_outputs = np.zeros(4)

            thrust, roll, pitch, yaw_rate = commands

            self.cf.commander.send_setpoint(roll, pitch, yaw_rate, int(thrust))
            self.logger.debug(f'Computed action: roll={roll}, pitch={pitch}, yaw_rate={yaw_rate}, thrust={thrust}')

            # Maintain timing
            elapsed_loop = time.time() - loop_start
            if elapsed_loop < interval:
                time.sleep(interval - elapsed_loop)

        self.cf.commander.send_stop_setpoint()

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
    controller = CrazyflieController()
    try:
        controller.start()
        while controller.is_connected:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.stop()
        print("Interrupted by user. Shutting down.")


if __name__ == '__main__':
    main()
