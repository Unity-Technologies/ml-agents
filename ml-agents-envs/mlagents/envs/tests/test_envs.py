import unittest.mock as mock
import pytest

import numpy as np

from mlagents.envs import (
    UnityEnvironment,
    UnityEnvironmentException,
    UnityActionException,
    AgentInfo,
)
from mlagents.envs.mock_communicator import MockCommunicator


@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_handles_bad_filename(get_communicator):
    with pytest.raises(UnityEnvironmentException):
        UnityEnvironment(" ")


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_initialization(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    with pytest.raises(UnityActionException):
        env.step([0])
    assert env.brain_names[0] == "RealFakeBrain"
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_reset(mock_communicator, mock_launcher):
    communicator = MockCommunicator(discrete_action=False, visual_inputs=0)
    mock_communicator.return_value = communicator
    env = UnityEnvironment(" ")
    brain = env.brains["RealFakeBrain"]
    agent_infos = env.reset()
    env.close()
    assert not env.global_done
    assert isinstance(agent_infos, list)
    assert len(agent_infos) == communicator.num_agents
    for agent_idx, agent_info in enumerate(agent_infos):
        assert isinstance(agent_infos[agent_idx], AgentInfo)
        assert isinstance(agent_infos[agent_idx].visual_observations, list)
        assert isinstance(agent_infos[agent_idx].vector_observations, np.ndarray)
        assert (
            len(agent_infos[agent_idx].visual_observations)
            == brain.number_visual_observations
        )
        assert (
            len(agent_infos[agent_idx].vector_observations)
            == brain.vector_observation_space_size
            * brain.num_stacked_vector_observations
        )


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_step(mock_communicator, mock_launcher):
    communicator = MockCommunicator(discrete_action=False, visual_inputs=0)
    mock_communicator.return_value = communicator
    env = UnityEnvironment(" ")
    brain = env.brains["RealFakeBrain"]
    agent_infos = env.reset()
    agent_infos = env.step([0] * brain.vector_action_space_size[0] * len(agent_infos))
    with pytest.raises(UnityActionException):
        env.step([0])
    agent_infos = env.step([-1] * brain.vector_action_space_size[0] * len(agent_infos))
    with pytest.raises(UnityActionException):
        env.step([0] * brain.vector_action_space_size[0] * len(agent_infos))
    env.close()
    assert env.global_done
    assert isinstance(agent_infos, list)
    assert isinstance(agent_infos[0], AgentInfo)
    assert agent_infos[0].brain_name == "RealFakeBrain"
    assert isinstance(agent_infos[0].visual_observations, list)
    assert isinstance(agent_infos[0].vector_observations, np.ndarray)
    assert len(agent_infos[0].visual_observations) == brain.number_visual_observations
    assert len(agent_infos) == communicator.num_agents
    assert (
        len(agent_infos[0].vector_observations)
        == brain.vector_observation_space_size * brain.num_stacked_vector_observations
    )

    assert not agent_infos[0].local_done
    assert agent_infos[2].local_done


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_close(mock_communicator, mock_launcher):
    comm = MockCommunicator(discrete_action=False, visual_inputs=0)
    mock_communicator.return_value = comm
    env = UnityEnvironment(" ")
    assert env._loaded
    env.close()
    assert not env._loaded
    assert comm.has_been_closed


if __name__ == "__main__":
    pytest.main()
