import gymnasium as gym
import numpy as np
import torch

import phoenix_drone_simulation
import time
from modules.env import DroneCircleBulletAttitudeEnv
from phoenix_drone_simulation.utils.utils import load_actor_critic_and_env_from_disk


def main():
    """
    Main function to test the environment
    """
    env = DroneCircleBulletAttitudeEnv(render_mode="human")
    model_path = "C:/Users/the_3/DroneRL/modules/phoenix-pybullet/saves/DroneCircleBulletEnv-v0/ppo/AttitudeMod/seed_65025"
    ac, _ = load_actor_critic_and_env_from_disk(model_path)

    done = False
    x, _ = env.reset()
    while not done:
        obs = torch.as_tensor(x, dtype=torch.float32)
        action, *_ = ac(obs)
        # current action is trust, roll, pitch, yaw_rate
        # update to roll, pitch, yaw_rate, trust using numpy
        # action = np.array([action[1], action[2], action[3], action[0]])
        # print size and content of the action space
        print(f'Action space size: {env.action_space.shape}')
        print(f'Action space : {env.action_space}')
        print(f'Action space content: {action}')


        x, reward, terminated, truncated, info = env.step(action)

        # print size and content of the observation space
        print(f'Observation space size: {env.observation_space.shape}')
        print(f'Observation space : {env.observation_space}')
        print(f'Observation space content: {x}')
        done = terminated or truncated
        time.sleep(0.05)


if __name__ == '__main__':
    main()
