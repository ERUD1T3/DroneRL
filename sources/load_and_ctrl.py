import logging
import threading
import time
from typing import Any, Dict, List, Tuple

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
latest_obs = None
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


def log_data(timestamp: float, data: Dict[str, Any], logconf: LogConfig) -> None:
    """
    Callback function to handle the incoming log data.
    """
    global latest_obs
    # Log the received data
    logger.info(f"[{timestamp}] {logconf.name} Received data: {data}")

    # Update the latest observations
    with obs_lock:
        # Construct the observation vector as expected by your model
        obs_list = [
            data.get('kalman.stateX', 0.0),
            data.get('kalman.stateY', 0.0),
            data.get('kalman.stateZ', 0.0),
            data.get('kalman.statePX', 0.0),
            data.get('kalman.statePY', 0.0),
            data.get('kalman.statePZ', 0.0),
            # Add more elements if your model expects them
        ]
        # Convert to numpy array
        obs = np.array(obs_list, dtype=np.float32)
        # Pad the observation vector with zeros to reach length 40
        if len(obs) < 40:
            obs = np.pad(obs, (0, 40 - len(obs)), 'constant')
        else:
            # If obs is longer than 40 elements, truncate it
            obs = obs[:40]
        # Update the latest observations
        latest_obs = obs


def start_logging(cf: Crazyflie) -> None:
    """
    Starts logging of the desired variables from the Crazyflie.
    """
    variables: List[Tuple[str, str]] = [
        # Position estimates
        ('kalman.stateX', 'float'),
        ('kalman.stateY', 'float'),
        ('kalman.stateZ', 'float'),
        # Velocity estimates
        ('kalman.statePX', 'float'),
        ('kalman.statePY', 'float'),
        ('kalman.statePZ', 'float'),
        # Add more variables as needed by your model
    ]

    # Maximum variables per log configuration
    max_vars_per_logconf = 10

    # Split variables into chunks
    variable_chunks = [variables[i:i + max_vars_per_logconf] for i in range(0, len(variables), max_vars_per_logconf)]

    for idx, var_chunk in enumerate(variable_chunks):
        log_conf_name = f'StateEstimation_{idx}'
        log_conf = LogConfig(name=log_conf_name, period_in_ms=10)  # 100 Hz logging

        # Add variables to log configuration
        for var_name, var_type in var_chunk:
            try:
                log_conf.add_variable(var_name, var_type)
            except KeyError as e:
                logger.error(f"Variable '{var_name}' not found in TOC. Ensure the required hardware is connected.")
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


def send_attitude_commands(cf: Crazyflie) -> None:
    """
    Sends attitude commands to the Crazyflie drone at 100 Hz.
    Makes the drone take off first before activating the model 3 seconds afterward.
    """
    global latest_obs

    # Unlock the safety lock
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    logger.info("Starting to send attitude commands at 100 Hz...")
    interval: float = 0.01  # 10 ms interval for 100 Hz
    takeoff_duration = 3.0  # Take off for 3 seconds

    # Record the start time of the function
    function_start_time = time.time()

    while is_connected:
        loop_start_time = time.time()

        elapsed_time = time.time() - function_start_time

        if elapsed_time < takeoff_duration:
            # Takeoff phase
            # Define the takeoff thrust value
            thrust = 37000  # Adjust this value as needed for takeoff
            roll = 0.0
            pitch = 0.0
            yaw_rate = 0.0

            logger.debug(f"Takeoff phase: time elapsed {elapsed_time:.2f}s, thrust={thrust}")
        else:
            # Control phase using the model
            # Retrieve the latest observations
            with obs_lock:
                obs = latest_obs

            if obs is not None:
                # Convert observations to torch tensor
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

                # Compute action using the trained model
                with torch.no_grad():
                    action, *_ = ac(obs_tensor)
                    # action = action.numpy()

                # Map action to roll, pitch, yaw_rate, thrust
                # Assuming action = [thrust, roll, pitch, yaw_rate]
                # Scale the actions to match Crazyflie's expected input ranges

                # Thrust scaling from [-1, 1] to [10001, 60000]
                thrust = (action[0] + 1) / 2 * (60000 - 10001) + 10001
                thrust = np.clip(thrust, 10001, 60000)

                # Roll and pitch scaling from [-1, 1] to [-30, 30] degrees
                max_roll_pitch = 30.0  # degrees
                roll = action[1] * max_roll_pitch
                pitch = action[2] * max_roll_pitch

                # Yaw rate scaling from [-1, 1] to [-200, 200] degrees per second
                max_yaw_rate = 200.0  # degrees per second
                yaw_rate = action[3] * max_yaw_rate

                # Ensure the values are within safe limits
                roll = np.clip(roll, -max_roll_pitch, max_roll_pitch)
                pitch = np.clip(pitch, -max_roll_pitch, max_roll_pitch)
                yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)

                logger.debug(f"Computed action: roll={roll}, pitch={pitch}, yaw_rate={yaw_rate}, thrust={thrust}")
            else:
                # If no observations are available yet, send zero commands
                thrust = 0
                roll = 0.0
                pitch = 0.0
                yaw_rate = 0.0
                logger.debug("No observations available yet; sending zero commands.")

        # Send the setpoint to the Crazyflie
        cf.commander.send_setpoint(roll, pitch, yaw_rate, int(thrust))
        logger.debug(f"Sent command: roll={roll}, pitch={pitch}, yaw_rate={yaw_rate}, thrust={thrust}")

        # Calculate the elapsed time and sleep for the remaining time to maintain 100 Hz
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
