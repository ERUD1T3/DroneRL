import logging
import threading
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

# Import your model loading function
from phoenix_drone_simulation.utils.utils import load_actor_critic_and_env_from_disk

# Configure the logger to write to a file with DEBUG level
logging.basicConfig(
    filename='crazyflie.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Global flag to control the main loop
is_connected: bool = True

# Global variable for the Crazyflie instance
cf: Crazyflie

# Global variables for observations and threading lock
latest_obs: Optional[np.ndarray] = None
last_model_outputs: np.ndarray = np.zeros(4, dtype=np.float32)  # Store raw model outputs
obs_lock = threading.Lock()


def connect_callback(uri: str) -> None:
    """
    Called when the Crazyflie is connected.
    """
    logger.info(f"Connected to {uri}")

    # Start logging data at 100 Hz
    start_logging(cf)

    # Start sending attitude commands at 100 Hz in a separate thread
    threading.Thread(target=send_attitude_commands, args=(cf,), daemon=True).start()


def disconnect_callback(uri: str) -> None:
    """
    Called when the Crazyflie is disconnected.
    """
    logger.info(f"Disconnected from {uri}")
    global is_connected
    is_connected = False


def connection_failed_callback(uri: str, msg: str) -> None:
    """
    Called when the Crazyflie connection fails.
    """
    logger.error(f"Connection to {uri} failed: {msg}")
    global is_connected
    is_connected = False


def connection_lost_callback(uri: str, msg: str) -> None:
    """
    Called when the Crazyflie connection is lost.
    """
    logger.warning(f"Connection to {uri} lost: {msg}")
    global is_connected
    is_connected = False


def start_logging(cf: Crazyflie) -> None:
    """
    Starts logging of variables required for the 20-dimensional observation space from the Crazyflie.

    Observation structure per timestep (20D):
    - Position (3D): kalman.stateX, kalman.stateY, kalman.stateZ
    - Quaternions (4D): kalman.q0, kalman.q1, kalman.q2, kalman.q3
    - Linear velocities (3D): kalman.statePX, kalman.statePY, kalman.statePZ
    - Angular velocities (3D): gyro.x, gyro.y, gyro.z
    - Position error (3D): calculated from position and target
    - Previous model outputs (4D): stored in last_model_outputs

    Args:
        cf (Crazyflie): The Crazyflie instance to configure logging for
    """
    variables: List[Tuple[str, str]] = [
        # Position estimates (x, y, z)
        ('kalman.stateX', 'float'),
        ('kalman.stateY', 'float'),
        ('kalman.stateZ', 'float'),

        # Quaternions from Kalman filter (q0, q1, q2, q3)
        ('kalman.q0', 'float'),
        ('kalman.q1', 'float'),
        ('kalman.q2', 'float'),
        ('kalman.q3', 'float'),

        # Linear velocities (vx, vy, vz)
        ('kalman.statePX', 'float'),
        ('kalman.statePY', 'float'),
        ('kalman.statePZ', 'float'),

        # Angular velocities from gyro (wx, wy, wz)
        ('gyro.x', 'float'),
        ('gyro.y', 'float'),
        ('gyro.z', 'float'),
    ]

    # Maximum variables per log configuration
    max_vars_per_logconf = 7

    # Split variables into chunks since Crazyflie has a limit on variables per config
    variable_chunks = [variables[i:i + max_vars_per_logconf]
                       for i in range(0, len(variables), max_vars_per_logconf)]

    for idx, var_chunk in enumerate(variable_chunks):
        log_conf_name = f'StateEstimation_{idx}'
        # Set to 10ms period for 100 Hz logging frequency
        log_conf = LogConfig(name=log_conf_name, period_in_ms=10)

        # Add variables to log configuration
        for var_name, var_type in var_chunk:
            try:
                log_conf.add_variable(var_name, var_type)
            except KeyError as e:
                logger.error(f"Variable '{var_name}' not found in TOC. "
                             f"Error: {e}. "
                             f"Ensure firmware has kalman estimator enabled.")
                continue

        # Add the log configuration to the Crazyflie
        try:
            cf.log.add_config(log_conf)
        except AttributeError as e:
            logger.error(f"Could not add log configuration '{log_conf_name}': {e}")
            continue

        # Register the callback function
        log_conf.data_received_cb.add_callback(log_data)

        # Start logging
        try:
            log_conf.start()
            logger.info(f"Started logging '{log_conf_name}' at 100 Hz.")
        except AttributeError as e:
            logger.error(f"Could not start log configuration '{log_conf_name}': {e}")
        except KeyError as e:
            logger.error(f"Could not start logging due to KeyError: {e}")
            logger.error("Ensure that all requested variables are available.")

        # Brief delay between adding configs to ensure stability
        time.sleep(0.1)


def log_data(timestamp: float, data: Dict[str, Any], logconf: LogConfig) -> None:
    """
    Constructs the 40-dimensional observation space for the neural network.
    The observation space consists of two consecutive timesteps, each with 20 dimensions.

    Observation structure (40D total):
    Current timestep (first 20D):
        - Position (3D): [x, y, z]
        - Quaternions (4D): [q0, q1, q2, q3]
        - Linear velocities (3D): [x_dot, y_dot, z_dot]
        - Angular velocities (3D): [roll_dot, pitch_dot, yaw_dot]
        - Position error (3D): [error_x, error_y, error_z]
        - Previous model outputs (4D): [out1, out2, out3, out4]

    Previous timestep (last 20D):
        - Same structure as current timestep

    Args:
        timestamp (float): Current timestamp
        data (Dict[str, Any]): Dictionary containing sensor data
        logconf (LogConfig): Logging configuration

    Note:
        Updates the global latest_obs variable with thread safety
    """
    global latest_obs, last_model_outputs
    static_obs = None

    with obs_lock:
        # Store current state as previous state if it exists
        if latest_obs is not None:
            static_obs = latest_obs.copy()

        # Construct current timestep observation (20 dimensions)
        current_obs = []

        # 1. Position from Kalman filter (3D)
        current_obs.extend([
            data.get('kalman.stateX', 0.0),
            data.get('kalman.stateY', 0.0),
            data.get('kalman.stateZ', 0.0)
        ])

        # 2. Quaternions from Kalman filter (4D)
        current_obs.extend([
            data.get('kalman.q0', 0.0),
            data.get('kalman.q1', 0.0),
            data.get('kalman.q2', 0.0),
            data.get('kalman.q3', 0.0)
        ])

        # 3. Linear velocities from Kalman filter (3D)
        current_obs.extend([
            data.get('kalman.statePX', 0.0),
            data.get('kalman.statePY', 0.0),
            data.get('kalman.statePZ', 0.0)
        ])

        # 4. Angular velocities (gyro data) (3D)
        # Note: getting raw gyro data, might need scaling based on your model's expectations
        current_obs.extend([
            data.get('gyro.x', 0.0),
            data.get('gyro.y', 0.0),
            data.get('gyro.z', 0.0)
        ])

        # 5. Position error relative to circle trajectory (3D)
        alpha: float = 2.0 * np.pi / 3.0
        target_x: float = 0.25 * (1.0 - np.cos(timestamp * alpha))
        target_y: float = 0.25 * np.sin(timestamp * alpha)
        target_z: float = 1.0

        current_pos = np.array([
            data.get('kalman.stateX', 0.0),
            data.get('kalman.stateY', 0.0),
            data.get('kalman.stateZ', 0.0)
        ])
        target_pos = np.array([target_x, target_y, target_z])
        error = target_pos - current_pos
        current_obs.extend(error)

        # 6. Previous model outputs (4D)
        # These are the raw outputs from the neural network
        current_obs.extend(last_model_outputs)

        # Convert to numpy array
        current_obs = np.array(current_obs, dtype=np.float32)

        # Combine current and previous observations
        # Note: Current obs goes first in the 40D vector
        if static_obs is not None:
            # Full 40D observation: [current_obs (20D), previous_obs (20D)]
            latest_obs = np.concatenate([static_obs[:20], current_obs])
        else:
            # If no previous observation exists, duplicate current
            latest_obs = np.concatenate([current_obs, current_obs])

        logger.debug(f"Observation shape: {latest_obs.shape}")
        logger.debug(f"Current observation: {latest_obs}")


def send_attitude_commands(cf: Crazyflie) -> None:
    """
    Sends attitude commands to the Crazyflie drone at 100 Hz.
    Makes the drone take off first before activating the model 3 seconds afterward.
    Updates the last_model_outputs with the raw network outputs.
    """
    global latest_obs, last_model_outputs

    # Unlock the safety lock
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    logger.info("Starting to send attitude commands at 100 Hz...")
    interval: float = 0.01  # 10 ms interval for 100 Hz
    takeoff_duration: float = 3.0  # Take off for 3 seconds

    function_start_time = time.time()

    while is_connected:
        loop_start_time = time.time()
        elapsed_time = time.time() - function_start_time

        if elapsed_time < takeoff_duration:
            # Takeoff phase
            thrust = 45000
            roll = 0.0
            pitch = 0.0
            yaw_rate = 0.0
            last_model_outputs = np.zeros(4)  # Reset during takeoff
            logger.debug(f"Takeoff phase: time elapsed {elapsed_time:.2f}s, thrust={thrust}")
        else:
            # Control phase using the model
            with obs_lock:
                obs = latest_obs

            if obs is not None:
                # Convert observations to torch tensor
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

                # Compute action using the trained model
                with torch.no_grad():
                    action, *_ = ac(obs_tensor)
                    # Store raw model outputs
                    last_model_outputs = action

                # Map action to roll, pitch, yaw_rate, thrust
                thrust = action[0] * (60000 - 10001) + 10001
                thrust = np.clip(thrust, 10001, 60000)

                max_roll_pitch = 30.0
                roll = action[1] * max_roll_pitch
                pitch = action[2] * max_roll_pitch

                max_yaw_rate = 200.0
                yaw_rate = action[3] * max_yaw_rate

                roll = np.clip(roll, -max_roll_pitch, max_roll_pitch)
                pitch = np.clip(pitch, -max_roll_pitch, max_roll_pitch)
                yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)

                logger.debug(f"Computed action: roll={roll}, pitch={pitch}, yaw_rate={yaw_rate}, thrust={thrust}")
            else:
                thrust = 0
                roll = 0.0
                pitch = 0.0
                yaw_rate = 0.0
                last_model_outputs = np.zeros(4)
                logger.debug("No observations available yet; sending zero commands.")

        # Send the setpoint to the Crazyflie
        cf.commander.send_setpoint(roll, pitch, yaw_rate, int(thrust))
        logger.debug(f"Sent command: roll={roll}, pitch={pitch}, yaw_rate={yaw_rate}, thrust={thrust}")

        # Maintain 100 Hz
        elapsed_loop_time = time.time() - loop_start_time
        sleep_time = interval - elapsed_loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.warning("Command sending is lagging behind the desired rate.")

    # Stop the motors when disconnected
    cf.commander.send_stop_setpoint()
    logger.info("Stopped sending attitude commands.")


def main() -> None:
    """
    Main function to connect to the Crazyflie drone, send attitude commands,
    and receive sensor observations at 100 Hz.
    """
    global is_connected, cf, ac

    # Initialize the low-level drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)

    # Load the trained model
    model_path = "C:/Users/the_3/DroneRL/modules/phoenix-pybullet/saves/DroneCircleBulletEnv-v0/ppo/AttitudeMod/seed_65025"
    ac, _ = load_actor_critic_and_env_from_disk(model_path)
    logger.info("Loaded trained model successfully.")

    # The URI of the Crazyflie to connect to
    URI: str = 'radio://0/80/2M'

    # Create a Crazyflie object
    cf = Crazyflie()

    # Register the callback functions
    cf.connected.add_callback(connect_callback)
    cf.disconnected.add_callback(disconnect_callback)
    cf.connection_failed.add_callback(connection_failed_callback)
    cf.connection_lost.add_callback(connection_lost_callback)

    # Open the link to the Crazyflie
    logger.info(f"Connecting to {URI}")
    cf.open_link(URI)

    # Keep the script running to receive callbacks
    try:
        while is_connected:
            time.sleep(1)
    except KeyboardInterrupt:
        # Close the link gracefully when interrupted
        cf.close_link()
        is_connected = False
        logger.info("Interrupted by user. Shutting down.")

if __name__ == '__main__':
    main()
