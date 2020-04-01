import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)


def main():
    env_name = "Project/testPlayer"
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=env_name,
        side_channels=[engine_configuration_channel],
        args=["-logFile", "-"],
    )

    try:
        # Reset the environment
        env.reset()

        # Set the default brain to work with
        group_name = env.get_agent_groups()[0]
        group_spec = env.get_agent_group_spec(group_name)

        # Set the time scale of the engine
        engine_configuration_channel.set_configuration_parameters(time_scale=3.0)

        # Get the state of the agents
        step_result = env.get_step_result(group_name)

        # Examine the number of observations per Agent
        print("Number of observations : ", len(group_spec.observation_shapes))

        # Is there a visual observation ?
        vis_obs = any(len(shape) == 3 for shape in group_spec.observation_shapes)
        print("Is there a visual observation ?", vis_obs)

        # Examine the state space for the first observation for the first agent
        print("First Agent observation looks like: \n{}".format(step_result.obs[0][0]))

        for _episode in range(10):
            env.reset()
            step_result = env.get_step_result(group_name)
            done = False
            episode_rewards = 0
            while not done:
                if group_spec.is_action_continuous():
                    action = np.random.randn(
                        step_result.n_agents(), group_spec.action_size
                    )

                elif group_spec.is_action_discrete():
                    branch_size = group_spec.discrete_action_branches
                    action = np.column_stack(
                        [
                            np.random.randint(
                                0, branch_size[i], size=(step_result.n_agents())
                            )
                            for i in range(len(branch_size))
                        ]
                    )
                else:
                    # Should never happen
                    action = None
                env.set_actions(group_name, action)
                env.step()
                step_result = env.get_step_result(group_name)
                episode_rewards += step_result.reward[0]
                done = step_result.done[0]
            print("Total reward this episode: {}".format(episode_rewards))
    finally:
        env.close()


if __name__ == "__main__":
    main()
