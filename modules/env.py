from phoenix_drone_simulation.envs.circle import DroneCircleBaseEnv


class DroneCircleBulletAttitudeEnv(DroneCircleBaseEnv):
    def __init__(self,
                 aggregate_phy_steps: int = 2,  # sub-steps used to calculate motor dynamics
                 control_mode: str = 'Attitude',  # use one of: PWM, Attitude, AttitudeRate
                 **kwargs):
        super(DroneCircleBulletAttitudeEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet',
            physics='PyBulletPhysics',
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )
