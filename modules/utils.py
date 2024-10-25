import numpy as np


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.

    Args:
        roll (float): Rotation angle around the X-axis in radians.
        pitch (float): Rotation angle around the Y-axis in radians.
        yaw (float): Rotation angle around the Z-axis in radians.

    Returns:
        np.ndarray: A NumPy array representing the quaternion [q0, q1, q2, q3],
                    where q0 is the scalar (real) part and [q1, q2, q3] are the vector (imaginary) parts.

    The quaternion is computed using the following formula:

        q0 = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
        q1 = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
        q2 = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
        q3 = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)

    Example:
        >>> roll = np.radians(30)    # 30 degrees in radians
        >>> pitch = np.radians(45)   # 45 degrees in radians
        >>> yaw = np.radians(60)     # 60 degrees in radians
        >>> quaternion = euler_to_quaternion(roll, pitch, yaw)
        >>> print(quaternion)
        [0.82236317 0.20056212 0.39190384 0.36042341]
    """
    # convert degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Compute the half angles
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    # Compute trigonometric functions of half angles
    cr = np.cos(half_roll)  # Cosine of half roll
    sr = np.sin(half_roll)  # Sine of half roll
    cp = np.cos(half_pitch)  # Cosine of half pitch
    sp = np.sin(half_pitch)  # Sine of half pitch
    cy = np.cos(half_yaw)  # Cosine of half yaw
    sy = np.sin(half_yaw)  # Sine of half yaw

    # Compute quaternion components
    q0 = cr * cp * cy + sr * sp * sy  # Scalar (real) part
    q1 = sr * cp * cy - cr * sp * sy  # X component
    q2 = cr * sp * cy + sr * cp * sy  # Y component
    q3 = cr * cp * sy - sr * sp * cy  # Z component

    # Return the quaternion as a NumPy array
    return np.array([q0, q1, q2, q3])
