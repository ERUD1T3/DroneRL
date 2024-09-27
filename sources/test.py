import gymnasium as gym
import phoenix_drone_simulation
import time
from modules.env import DroneCircleBulletAttitudeEnv


def main():
    """
    Main function to test the environment
    """
    env = DroneCircleBulletAttitudeEnv(render_mode="human")

    while True:
        done = False
        x, _ = env.reset()
        while not done:
            random_action = env.action_space.sample()
            # print size and content of the action space
            print(f'Action space size: {env.action_space.shape}')
            print(f'Action space content: {random_action}')

            x, reward, terminated, truncated, info = env.step(random_action)

            # print size and content of the observation space
            print(f'Observation space size: {env.observation_space.shape}')
            print(f'Observation space content: {x}')
            done = terminated or truncated
            time.sleep(0.05)


if __name__ == '__main__':
    main()
