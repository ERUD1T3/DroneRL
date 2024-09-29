import time

import cflib.crtp
from cflib.crazyflie import Crazyflie


def simple_sequence(cf):
    # Unlock the safety lock by sending a zero setpoint
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    thrust = 20000  # Adjust thrust as needed (range: 10001 - 60000)
    roll = 0.0
    pitch = 0.0
    yaw_rate = 0.0

    # Send the desired setpoints
    for _ in range(100):
        cf.commander.send_setpoint(roll, pitch, yaw_rate, thrust)
        time.sleep(0.1)

    # Stop the motors after the sequence
    cf.commander.send_stop_setpoint()


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)

    # The URI of the Crazyflie to connect to
    URI = 'radio://0/80/2M'

    # Create a Crazyflie object
    cf = Crazyflie(rw_cache='./cache')


    # Connect and run the sequence
    def connect_and_run(uri):
        print(f"Connected to {uri}")
        simple_sequence(cf)


    # Register the connection callback
    cf.connected.add_callback(connect_and_run)

    # Connect to the Crazyflie
    print(f"Connecting to {URI}")
    cf.open_link(URI)

    # Keep the script running until the sequence is done
    try:
        # Wait for the sequence to finish
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        # Close the link when done
        cf.close_link()
        print("Disconnected from Crazyflie")
