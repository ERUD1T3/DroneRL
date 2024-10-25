import logging
import threading
import time
from typing import Any, Dict, List, Tuple

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

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


def connect_callback(uri: str) -> None:
    """
    Called when the Crazyflie is connected.

    Args:
        uri (str): The URI of the connected Crazyflie.
    """
    logger.info(f"Connected to {uri}")

    # Start logging data at 100 Hz
    start_logging(cf)

    # Start sending attitude commands at 100 Hz in a separate thread
    threading.Thread(target=send_attitude_commands, args=(cf,), daemon=True).start()


def disconnect_callback(uri: str) -> None:
    """
    Called when the Crazyflie is disconnected.

    Args:
        uri (str): The URI of the disconnected Crazyflie.
    """
    logger.info(f"Disconnected from {uri}")
    global is_connected
    is_connected = False


def connection_failed_callback(uri: str, msg: str) -> None:
    """
    Called when the Crazyflie connection fails.

    Args:
        uri (str): The URI of the Crazyflie.
        msg (str): The error message.
    """
    logger.error(f"Connection to {uri} failed: {msg}")
    global is_connected
    is_connected = False


def connection_lost_callback(uri: str, msg: str) -> None:
    """
    Called when the Crazyflie connection is lost.

    Args:
        uri (str): The URI of the Crazyflie.
        msg (str): The error message.
    """
    logger.warning(f"Connection to {uri} lost: {msg}")
    global is_connected
    is_connected = False


def log_data(timestamp: float, data: Dict[str, Any], logconf: LogConfig) -> None:
    """
    Callback function to handle the incoming log data.

    Args:
        timestamp (float): The timestamp of the data.
        data (Dict[str, Any]): The logged data.
        logconf (LogConfig): The log configuration.
    """
    # Log the received data
    logger.info(f"[{timestamp}] {logconf.name} Received data: {data}")


def start_logging(cf: Crazyflie) -> None:
    """
    Starts logging of the desired variables from the Crazyflie.
    NOTE: test against results from client and the result from the python script
    are somewhat accurate.

    Args:
        cf (Crazyflie): The Crazyflie instance to log data from.
    """
    # Define the variables to log
    variables: List[Tuple[str, str]] = [
        # # Orientation angles
        # ('stabilizer.roll', 'float'),
        # ('stabilizer.pitch', 'float'),
        # ('stabilizer.yaw', 'float'),
        #
        # # Angular velocities # commented out to reduce the number of variables
        # # ('gyro.x', 'float'),
        # # ('gyro.y', 'float'),
        # # ('gyro.z', 'float'),
        #
        # # x, y, z position
        # ('stateEstimate.x', 'float'),
        # ('stateEstimate.y', 'float'),
        # ('stateEstimate.z', 'float'),
        #
        # # EKF state
        # ('kalman.stateX', 'float'),
        # ('kalman.stateY', 'float'),
        # ('kalman.stateZ', 'float'),

        # Position estimates
        ('kalman.stateX', 'float'),
        ('kalman.stateY', 'float'),
        ('kalman.stateZ', 'float'),
        # Velocity estimates
        ('kalman.statePX', 'float'),
        ('kalman.statePY', 'float'),
        ('kalman.statePZ', 'float'),
        # Attitude as quaternion
        # ('kalman.q0', 'float'),
        # ('kalman.q1', 'float'),
        # ('kalman.q2', 'float'),
        # ('kalman.q3', 'float'),
        # Thrust
        ('stabilizer.thrust', 'uint16_t'),
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
    Sends attitude commands to the Crazyflie drone at 100 Hz and logs them.

    Args:
        cf (Crazyflie): The Crazyflie instance to send commands to.
    """
    # IMPORTANT: Unlock the safety lock by sending a zero setpoint
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    # Define the command parameters
    thrust: int = 15000  # Adjust as needed (range: 10001 - 60000)
    roll: float = 0.0  # Roll angle in degrees
    pitch: float = 0.0  # Pitch angle in degrees
    yaw_rate: float = 0.0  # Yaw rate in degrees per second

    logger.info("Starting to send attitude commands at 100 Hz...")
    # Calculate the time interval between commands (10 ms)
    interval: float = 0.01

    # log if is connected
    logger.info(f"Is connected: {is_connected}")

    # TODO: use MC to take off before sending commands

    # Continue sending commands until disconnected
    while is_connected:
        start_time = time.time()

        # Send the setpoint to the Crazyflie
        cf.commander.send_setpoint(roll, pitch, yaw_rate, thrust)
        # Log the command sent
        logger.debug(f"Sent command: roll={roll}, pitch={pitch}, yaw_rate={yaw_rate}, thrust={thrust}")

        # Calculate the elapsed time and sleep for the remaining time to maintain 100 Hz
        elapsed_time = time.time() - start_time
        sleep_time = interval - elapsed_time
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
    global is_connected, cf

    # Initialize the low-level drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)

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
