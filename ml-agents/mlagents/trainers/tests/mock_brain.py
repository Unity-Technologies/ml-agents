import unittest.mock as mock
import pytest
import numpy as np

from mlagents.trainers.buffer import Buffer


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


def create_mock_braininfo(
    num_agents=1,
    num_vector_observations=0,
    num_vis_observations=0,
    num_vector_acts=2,
    discrete=False,
):
    """
    Creates a mock BrainInfo with observations. Imitates constant
    vector/visual observations, rewards, dones, and agents.

    :int num_agents: Number of "agents" to imitate in your BrainInfo values.
    :int num_vector_observations: Number of "observations" in your observation space
    :int num_vis_observations: Number of "observations" in your observation space
    :int num_vector_acts: Number of actions in your action space
    :bool discrete: Whether or not action space is discrete
    """
    mock_braininfo = mock.Mock()

    mock_braininfo.return_value.visual_observations = num_vis_observations * [
        np.ones((num_agents, 84, 84, 3))
    ]
    mock_braininfo.return_value.vector_observations = np.array(
        num_agents * [num_vector_observations * [1]]
    )
    if discrete:
        mock_braininfo.return_value.previous_vector_actions = np.array(
            num_agents * [1 * [0.5]]
        )
        mock_braininfo.return_value.action_masks = np.array(
            num_agents * [num_vector_acts * [1.0]]
        )
    else:
        mock_braininfo.return_value.previous_vector_actions = np.array(
            num_agents * [num_vector_acts * [0.5]]
        )
    mock_braininfo.return_value.memories = np.ones((num_agents, 8))
    mock_braininfo.return_value.rewards = num_agents * [1.0]
    mock_braininfo.return_value.local_done = num_agents * [False]
    mock_braininfo.return_value.text_observations = num_agents * [""]
    mock_braininfo.return_value.agents = range(0, num_agents)
    return mock_braininfo()


def setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo):
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
    mock_env.return_value.reset.return_value = {"MockBrain": mock_braininfo}
    mock_env.return_value.step.return_value = {"MockBrain": mock_braininfo}


def simulate_rollout(env, policy, buffer_init_samples):
    brain_info_list = []
    for i in range(buffer_init_samples):
        brain_info_list.append(env.step()[env.brain_names[0]])
    buffer = create_buffer(brain_info_list, policy.brain, policy.sequence_length)
    return buffer


def create_buffer(brain_infos, brain_params, sequence_length):
    buffer = Buffer()
    # Make a buffer
    for idx, experience in enumerate(brain_infos):
        if idx > len(brain_infos) - 2:
            break
        current_brain_info = brain_infos[idx]
        next_brain_info = brain_infos[idx + 1]
        buffer[0].last_brain_info = current_brain_info
        buffer[0]["done"].append(next_brain_info.local_done[0])
        buffer[0]["rewards"].append(next_brain_info.rewards[0])
        for i in range(brain_params.number_visual_observations):
            buffer[0]["visual_obs%d" % i].append(
                current_brain_info.visual_observations[i][0]
            )
            buffer[0]["next_visual_obs%d" % i].append(
                current_brain_info.visual_observations[i][0]
            )
        if brain_params.vector_observation_space_size > 0:
            buffer[0]["vector_obs"].append(current_brain_info.vector_observations[0])
            buffer[0]["next_vector_in"].append(
                current_brain_info.vector_observations[0]
            )
        buffer[0]["actions"].append(next_brain_info.previous_vector_actions[0])
        buffer[0]["prev_action"].append(current_brain_info.previous_vector_actions[0])
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
