import argparse
import numpy as np

from gym_unity.envs import UnityEnv


def main(env_name):
    """
    Run the gym test using the specified environment
    :param env_name: Name of the Unity environment binary to launch
    """
    multi_env = UnityEnv(
        env_name, worker_id=1, use_visual=False, multiagent=True, no_graphics=True
    )

    try:
        # Examine environment parameters
        print(str(multi_env))

        # Reset the environment
        initial_observations = multi_env.reset()

        if len(multi_env.observation_space.shape) == 1:
            # Examine the initial vector observation
            print("Agent observations look like: \n{}".format(initial_observations[0]))

        for _episode in range(10):
            multi_env.reset()
            done = False
            episode_rewards = 0
            while not done:
                actions = [
                    multi_env.action_space.sample()
                    for agent in range(multi_env.number_agents)
                ]
                observations, rewards, dones, info = multi_env.step(actions)
                episode_rewards += np.mean(rewards)
                done = dones[0]
            print("Total reward this episode: {}".format(episode_rewards))
    finally:
        multi_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Project/testPlayer")
    args = parser.parse_args()
    main(args.env)
