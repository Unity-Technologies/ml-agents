import argparse

from gym_unity.envs import UnityEnv


def main(env_name):
    """
    Run the gym test using the specified environment
    :param env_name: Name of the Unity environment binary to launch
    """
    env = UnityEnv(env_name, worker_id=1, use_visual=False, no_graphics=True)

    try:
        # Examine environment parameters
        print(str(env))

        # Reset the environment
        initial_observations = env.reset()

        if len(env.observation_space.shape) == 1:
            # Examine the initial vector observation
            print("Agent observations look like: \n{}".format(initial_observations))

        for _episode in range(10):
            env.reset()
            done = False
            episode_rewards = 0
            while not done:
                actions = env.action_space.sample()
                obs, reward, done, _ = env.step(actions)
                episode_rewards += reward
            print("Total reward this episode: {}".format(episode_rewards))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Project/testPlayer")
    args = parser.parse_args()
    main(args.env)
