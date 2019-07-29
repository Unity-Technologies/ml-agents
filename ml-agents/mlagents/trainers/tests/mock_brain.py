import unittest.mock as mock
import numpy as np
from typing import List

from mlagents.trainers.buffer import Buffer
from mlagents.envs.env_manager import AgentStep
from mlagents.envs.brain import AgentInfo


def create_mock_brainparams(
    number_visual_observations=0,
    num_stacked_vector_observations=1,
    vector_action_space_type="continuous",
    vector_observation_space_size=3,
    vector_action_space_size=None,
):
    """
    Creates a mock BrainParameters object with parameters.
    """
    # Avoid using mutable object as default param
    if vector_action_space_size is None:
        vector_action_space_size = [2]
    mock_brain = mock.Mock()
    mock_brain.return_value.number_visual_observations = number_visual_observations
    mock_brain.return_value.num_stacked_vector_observations = (
        num_stacked_vector_observations
    )
    mock_brain.return_value.vector_action_space_type = vector_action_space_type
    mock_brain.return_value.vector_observation_space_size = (
        vector_observation_space_size
    )
    camrez = {"blackAndWhite": False, "height": 84, "width": 84}
    mock_brain.return_value.camera_resolutions = [camrez] * number_visual_observations
    mock_brain.return_value.vector_action_space_size = vector_action_space_size
    return mock_brain()


def create_mock_agentinfos(
    num_agents=1,
    num_vector_observations=0,
    num_vis_observations=0,
    num_vector_acts=2,
    discrete=False,
):
    """
    Creates mock AgentInfos with observations. Imitates constant
    vector/visual observations, rewards, dones, and agents.

    :int num_agents: Number of "agents" to imitate.
    :int num_vector_observations: Number of "observations" in your observation space
    :int num_vis_observations: Number of "observations" in your observation space
    :int num_vector_acts: Number of actions in your action space
    :bool discrete: Whether or not action space is discrete
    """
    agent_infos = []
    for i in range(num_agents):
        if discrete:
            previous_vector_actions = np.array([0.5])
            action_mask = np.array(num_vector_acts * [1.0])
        else:
            previous_vector_actions = np.array(num_vector_acts * [0.5])
            action_mask = None
        agent_info = AgentInfo(
            brain_name="MockBrain",
            visual_observation=num_vis_observations * [np.ones((84, 84, 3))],
            vector_observation=np.array(num_vector_observations * [1]),
            memory=[1] * 8,
            reward=1.0,
            local_done=False,
            text_observation="",
            id=f"{i}",
            previous_vector_actions=previous_vector_actions,
            action_mask=action_mask,
        )
        agent_infos.append(agent_info)
    return agent_infos


def setup_mock_unityenvironment(mock_env, mock_brain, mock_agentinfos):
    """
    Takes a mock UnityEnvironment and adds the appropriate properties, defined by the mock
    BrainParameters and BrainInfo.

    :Mock mock_env: A mock UnityEnvironment, usually empty.
    :Mock mock_brain: A mock Brain object that specifies the params of this environment.
    :Mock mock_braininfo: A mock BrainInfo object that will be returned at each step and reset.
    """
    mock_env.return_value.academy_name = "MockAcademy"
    mock_env.return_value.brains = {"MockBrain": mock_brain}
    mock_env.return_value.external_brain_names = ["MockBrain"]
    mock_env.return_value.brain_names = ["MockBrain"]
    mock_env.return_value.reset.return_value = mock_agentinfos
    mock_env.return_value.step.return_value = mock_agentinfos


def simulate_rollout(env, policy, buffer_init_samples):
    agent_info_list = []
    for i in range(buffer_init_samples):
        agent_info_list.extend(env.step())
    buffer = create_buffer(agent_info_list, policy.brain, policy.sequence_length)
    return buffer


def create_buffer(agent_infos, brain_params, sequence_length):
    buffer = Buffer()
    # Make a buffer
    for idx, agent_info in enumerate(agent_infos):
        if idx > len(agent_infos) - 2:
            break
        current_agent_info: AgentInfo = agent_infos[idx]
        next_agent_info: AgentInfo = agent_infos[idx + 1]
        buffer[0]["done"].append(next_agent_info.local_done)
        buffer[0]["rewards"].append(next_agent_info.reward)
        for i in range(brain_params.number_visual_observations):
            buffer[0]["visual_obs%d" % i].append(
                current_agent_info.visual_observations[i]
            )
            buffer[0]["next_visual_obs%d" % i].append(
                current_agent_info.visual_observations[i]
            )
        if brain_params.vector_observation_space_size > 0:
            buffer[0]["vector_obs"].append(current_agent_info.vector_observations)
            buffer[0]["next_vector_in"].append(current_agent_info.vector_observations)
        buffer[0]["actions"].append(next_agent_info.previous_vector_actions)
        buffer[0]["prev_action"].append(current_agent_info.previous_vector_actions)
        buffer[0]["masks"].append(1.0)
        buffer[0]["advantages"].append(1.0)
        buffer[0]["action_probs"].append(np.ones(buffer[0]["actions"][0].shape))
        buffer[0]["actions_pre"].append(np.ones(buffer[0]["actions"][0].shape))
        buffer[0]["random_normal_epsilon"].append(
            np.ones(buffer[0]["actions"][0].shape)
        )
        buffer[0]["action_mask"].append(np.ones(buffer[0]["actions"][0].shape))
        buffer[0]["memory"].append(np.ones(8))

    buffer.append_update_buffer(0, batch_size=None, training_length=sequence_length)
    return buffer
