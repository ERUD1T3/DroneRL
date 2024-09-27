import gymnasium as gym
import phoenix_drone_simulation
import time


def main():
    """
    Main function to test the environment
    """
    env = gym.make('DroneCircleBulletEnv-v0', render_mode="human")

    while True:
        done = False
        x, _ = env.reset()
        while not done:
            random_action = env.action_space.sample()
            x, reward, terminated, truncated, info = env.step(random_action)
            done = terminated or truncated
            time.sleep(0.05)


if __name__ == '__main__':
    main()
