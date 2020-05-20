import argparse

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


def test_run_environment(env_name):
    """
    Run the gym test using the specified environment
    :param env_name: Name of the Unity environment binary to launch
    """
    u_env = UnityEnvironment(env_name, worker_id=1, no_graphics=True)
    env = UnityToGymWrapper(u_env)

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


def test_closing(env_name):
    """
    Run the gym test and closes the environment multiple times
    :param env_name: Name of the Unity environment binary to launch
    """

    try:
        env1 = UnityToGymWrapper(
            UnityEnvironment(env_name, worker_id=1, no_graphics=True)
        )
        env1.close()
        env1 = UnityToGymWrapper(
            UnityEnvironment(env_name, worker_id=1, no_graphics=True)
        )
        env2 = UnityToGymWrapper(
            UnityEnvironment(env_name, worker_id=2, no_graphics=True)
        )
        env2.reset()
    finally:
        env1.close()
        env2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Project/testPlayer")
    args = parser.parse_args()
    test_run_environment(args.env)
    test_closing(args.env)
