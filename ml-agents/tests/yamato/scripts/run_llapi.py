import argparse
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)


def main(env_name):
    """
    Run the low-level API test using the specified environment
    :param env_name: Name of the Unity environment binary to launch
    """
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=env_name,
        side_channels=[engine_configuration_channel],
        no_graphics=True,
        args=["-logFile", "-"],
    )

    try:
        # Reset the environment
        env.reset()

        # Set the default brain to work with
        group_name = env.get_behavior_names()[0]
        group_spec = env.get_behavior_spec(group_name)

        # Set the time scale of the engine
        engine_configuration_channel.set_configuration_parameters(time_scale=3.0)

        # Get the state of the agents
        decision_steps, terminal_steps = env.get_steps(group_name)

        # Examine the number of observations per Agent
        print("Number of observations : ", len(group_spec.observation_shapes))

        # Is there a visual observation ?
        vis_obs = any(len(shape) == 3 for shape in group_spec.observation_shapes)
        print("Is there a visual observation ?", vis_obs)

        # Examine the state space for the first observation for the first agent
        print(
            "First Agent observation looks like: \n{}".format(decision_steps.obs[0][0])
        )

        for _episode in range(10):
            env.reset()
            decision_steps, terminal_steps = env.get_steps(group_name)
            done = False
            episode_rewards = 0
            tracked_agent = -1
            while not done:
                if group_spec.is_action_continuous():
                    action = np.random.randn(
                        len(decision_steps), group_spec.action_size
                    )

                elif group_spec.is_action_discrete():
                    branch_size = group_spec.discrete_action_branches
                    action = np.column_stack(
                        [
                            np.random.randint(
                                0, branch_size[i], size=(len(decision_steps))
                            )
                            for i in range(len(branch_size))
                        ]
                    )
                else:
                    # Should never happen
                    action = None
                if tracked_agent == -1 and len(decision_steps) > 1:
                    tracked_agent = decision_steps.agent_id[0]
                env.set_actions(group_name, action)
                env.step()
                decision_steps, terminal_steps = env.get_steps(group_name)
                done = False
                if tracked_agent in decision_steps:
                    episode_rewards += decision_steps[tracked_agent].reward
                if tracked_agent in terminal_steps:
                    episode_rewards += terminal_steps[tracked_agent].reward
                    done = True
            print("Total reward this episode: {}".format(episode_rewards))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Project/testPlayer")
    args = parser.parse_args()
    main(args.env)
