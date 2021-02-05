import argparse
import numpy as np

from mlagents_envs.environment import UnityEnvironment

EPSILON = 0.001


def test_run_environment(env_name):
    """
    Run the low-level API test of compressed sensors using the specified environment
    :param env_name: Name of the Unity environment binary to launch
    """
    env = UnityEnvironment(
        file_name=env_name, no_graphics=True, additional_args=["-logFile", "-"]
    )

    try:
        # Reset the environment
        env.reset()

        env.step()

        # Set the default brain to work with
        group_name = list(env.behavior_specs.keys())[0]

        # Get the state of the agents
        decision_steps, _ = env.get_steps(group_name)

        # One observation comes from compressed sensor while the other comes
        # from an uncompressed sensor
        obs_1 = decision_steps.obs[0][0, :, :, :]
        obs_2 = decision_steps.obs[0][1, :, :, :]

        diff = np.abs(obs_1 - obs_2)

        # make sure both are identical
        assert np.max(diff) < EPSILON

        # make sure an actual observation was collected
        assert np.max(obs_1) > EPSILON

        print("Observations were identical")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="artifacts/testPlayer")
    args = parser.parse_args()
    test_run_environment(args.env)
