import unittest.mock as mock
import pytest
import numpy as np


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
