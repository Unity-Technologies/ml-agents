import unittest.mock as mock
import pytest
import yaml
import mlagents.trainers.tests.mock_brain as mb
import numpy as np
from mlagents.trainers.rl_trainer import RLTrainer
from mlagents.trainers.tests.test_buffer import construct_fake_buffer


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        summary_path: "test/"
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


def create_mock_brain():
    mock_brain = mb.create_mock_brainparams(
        vector_action_space_type="continuous",
        vector_action_space_size=[2],
        vector_observation_space_size=8,
        number_visual_observations=1,
    )
    return mock_brain


def create_rl_trainer():
    mock_brainparams = create_mock_brain()
    trainer = RLTrainer(mock_brainparams, dummy_config(), True, 0)
    return trainer


def create_mock_all_brain_info(brain_info):
    return {"MockBrain": brain_info}


def create_mock_policy():
    mock_policy = mock.Mock()
    mock_policy.reward_signals = {}
    return mock_policy


@mock.patch("mlagents.trainers.rl_trainer.RLTrainer.add_policy_outputs")
@mock.patch("mlagents.trainers.rl_trainer.RLTrainer.add_rewards_outputs")
@pytest.mark.parametrize("num_vis_obs", [0, 1, 2], ids=["vec", "1 viz", "2 viz"])
def test_rl_trainer(add_policy_outputs, add_rewards_outputs, num_vis_obs):
    trainer = create_rl_trainer()
    trainer.policy = create_mock_policy()
    fake_action_outputs = {
        "action": [0.1, 0.1],
        "value_heads": {},
        "entropy": np.array([1.0]),
        "learning_rate": 1.0,
    }
    mock_braininfo = mb.create_mock_braininfo(
        num_agents=2,
        num_vector_observations=8,
        num_vector_acts=2,
        num_vis_observations=num_vis_obs,
    )
    trainer.add_experiences(
        create_mock_all_brain_info(mock_braininfo),
        create_mock_all_brain_info(mock_braininfo),
        fake_action_outputs,
    )

    # Remove one of the agents
    next_mock_braininfo = mb.create_mock_braininfo(
        num_agents=1,
        num_vector_observations=8,
        num_vector_acts=2,
        num_vis_observations=num_vis_obs,
    )
    brain_info = trainer.construct_curr_info(next_mock_braininfo)

    # assert construct_curr_info worked properly
    assert len(brain_info.agents) == 1
    assert len(brain_info.visual_observations) == num_vis_obs
    assert len(brain_info.vector_observations) == 1
    assert len(brain_info.previous_vector_actions) == 1

    # Test end episode
    trainer.end_episode()
    for agent_id in trainer.episode_steps:
        assert trainer.episode_steps[agent_id] == 0
        assert len(trainer.training_buffer[agent_id]["action"]) == 0
    for rewards in trainer.collected_rewards.values():
        for agent_id in rewards:
            assert rewards[agent_id] == 0


def test_clear_update_buffer():
    trainer = create_rl_trainer()
    trainer.training_buffer = construct_fake_buffer()
    trainer.training_buffer.append_update_buffer(2, batch_size=None, training_length=2)
    trainer.clear_update_buffer()
    for _, arr in trainer.training_buffer.update_buffer.items():
        assert len(arr) == 0
