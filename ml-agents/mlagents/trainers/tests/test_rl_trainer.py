from unittest import mock
import yaml
import mlagents.trainers.tests.mock_brain as mb
import numpy as np
from mlagents.trainers.rl_trainer import RLTrainer
from mlagents.trainers.tests.test_buffer import construct_fake_buffer


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
    trainer = RLTrainer(mock_brainparams.brain_name, dummy_config(), True, 0)
    return trainer


def create_mock_all_brain_info(brain_info):
    return {"MockBrain": brain_info}


def create_mock_policy():
    mock_policy = mock.Mock()
    mock_policy.reward_signals = {}
    mock_policy.retrieve_memories.return_value = np.zeros((1, 1), dtype=np.float32)
    mock_policy.retrieve_previous_action.return_value = np.zeros(
        (1, 1), dtype=np.float32
    )
    return mock_policy


def test_rl_trainer():
    trainer = create_rl_trainer()
    agent_id = "0"
    trainer.episode_steps[agent_id] = 3
    trainer.collected_rewards["extrinsic"] = {agent_id: 3}
    # Test end episode
    trainer.end_episode()
    for agent_id in trainer.episode_steps:
        assert trainer.episode_steps[agent_id] == 0
    for rewards in trainer.collected_rewards.values():
        for agent_id in rewards:
            assert rewards[agent_id] == 0


def test_clear_update_buffer():
    trainer = create_rl_trainer()
    trainer.update_buffer = construct_fake_buffer(0)
    trainer.clear_update_buffer()
    for _, arr in trainer.update_buffer.items():
        assert len(arr) == 0
