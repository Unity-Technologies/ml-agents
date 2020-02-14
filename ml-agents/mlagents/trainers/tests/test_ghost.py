import pytest

import numpy as np

import yaml

from mlagents.trainers.ghost.trainer import GhostTrainer
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        trainer: ppo
        batch_size: 32
        beta: 5.0e-3
        buffer_size: 512
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 3.0e-4
        max_steps: 5.0e4
        normalize: true
        num_epoch: 5
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        use_recurrent: false
        normalize: true
        memory_size: 8
        curiosity_strength: 0.0
        curiosity_enc_size: 1
        summary_path: test
        model_path: test
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        self_play:
            window: 5
            play_against_current_self_ratio: 0.5
            save_steps: 1000
            swap_steps: 1000
        """
    )


VECTOR_ACTION_SPACE = [1]
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 513
NUM_AGENTS = 12


@pytest.mark.parametrize("use_discrete", [True, False])
def test_load_and_set(dummy_config, use_discrete):
    mock_brain = mb.setup_mock_brain(
        use_discrete,
        False,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_params = dummy_config
    trainer = PPOTrainer(
        mock_brain.brain_name, 0, trainer_params, True, False, 0, "0", False
    )
    trainer.seed = 1
    policy = trainer.create_policy(mock_brain)
    trainer.seed = 20  # otherwise graphs are the same
    to_load_policy = trainer.create_policy(mock_brain)
    to_load_policy.init_load_weights()

    weights = policy.get_weights()
    load_weights = to_load_policy.get_weights()
    try:
        for w, lw in zip(weights, load_weights):
            np.testing.assert_array_equal(w, lw)
    except AssertionError:
        pass

    to_load_policy.load_weights(weights)
    load_weights = to_load_policy.get_weights()

    for w, lw in zip(weights, load_weights):
        np.testing.assert_array_equal(w, lw)


def test_process_trajectory(dummy_config):
    brain_params_team0 = BrainParameters(
        brain_name="test_brain?team=0",
        vector_observation_space_size=1,
        camera_resolutions=[],
        vector_action_space_size=[2],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )

    brain_name = BehaviorIdentifiers.from_name_behavior_id(
        brain_params_team0.brain_name
    ).brain_name

    brain_params_team1 = BrainParameters(
        brain_name="test_brain?team=1",
        vector_observation_space_size=1,
        camera_resolutions=[],
        vector_action_space_size=[2],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )
    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    ppo_trainer = PPOTrainer(brain_name, 0, dummy_config, True, False, 0, "0", False)
    trainer = GhostTrainer(ppo_trainer, brain_name, 0, dummy_config, True, "0")

    # first policy encountered becomes policy trained by wrapped PPO
    policy = trainer.create_policy(brain_params_team0)
    trainer.add_policy(brain_params_team0.brain_name, policy)
    trajectory_queue0 = AgentManagerQueue(brain_params_team0.brain_name)
    trainer.subscribe_trajectory_queue(trajectory_queue0)

    # Ghost trainer should ignore this queue because off policy
    policy = trainer.create_policy(brain_params_team1)
    trainer.add_policy(brain_params_team1.brain_name, policy)
    trajectory_queue1 = AgentManagerQueue(brain_params_team1.brain_name)
    trainer.subscribe_trajectory_queue(trajectory_queue1)

    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        vec_obs_size=1,
        num_vis_obs=0,
        action_space=[2],
    )
    trajectory_queue0.put(trajectory)
    trainer.advance()

    # Check that trainer put trajectory in update buffer
    assert trainer.trainer.update_buffer.num_experiences == 15

    trajectory_queue1.put(trajectory)
    trainer.advance()

    # Check that ghost trainer ignored off policy queue
    assert trainer.trainer.update_buffer.num_experiences == 15
    # Check that it emptied the queue
    assert trajectory_queue1.empty()


def test_publish_queue(dummy_config):
    brain_params_team0 = BrainParameters(
        brain_name="test_brain?team=0",
        vector_observation_space_size=8,
        camera_resolutions=[],
        vector_action_space_size=[1],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )

    brain_name = BehaviorIdentifiers.from_name_behavior_id(
        brain_params_team0.brain_name
    ).brain_name

    brain_params_team1 = BrainParameters(
        brain_name="test_brain?team=1",
        vector_observation_space_size=8,
        camera_resolutions=[],
        vector_action_space_size=[1],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )
    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    ppo_trainer = PPOTrainer(brain_name, 0, dummy_config, True, False, 0, "0", False)
    trainer = GhostTrainer(ppo_trainer, brain_name, 0, dummy_config, True, "0")

    # First policy encountered becomes policy trained by wrapped PPO
    # This queue should remain empty after swap snapshot
    policy = trainer.create_policy(brain_params_team0)
    trainer.add_policy(brain_params_team0.brain_name, policy)
    policy_queue0 = AgentManagerQueue(brain_params_team0.brain_name)
    trainer.publish_policy_queue(policy_queue0)

    # Ghost trainer should use this queue for ghost policy swap
    policy = trainer.create_policy(brain_params_team1)
    trainer.add_policy(brain_params_team1.brain_name, policy)
    policy_queue1 = AgentManagerQueue(brain_params_team1.brain_name)
    trainer.publish_policy_queue(policy_queue1)

    # check ghost trainer swap pushes to ghost queue and not trainer
    assert policy_queue0.empty() and policy_queue1.empty()
    trainer._swap_snapshots()
    assert policy_queue0.empty() and not policy_queue1.empty()
    # clear
    policy_queue1.get_nowait()

    mock_brain = mb.setup_mock_brain(
        False,
        False,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, mock_brain)
    # Mock out reward signal eval
    buffer["extrinsic_rewards"] = buffer["environment_rewards"]
    buffer["extrinsic_returns"] = buffer["environment_rewards"]
    buffer["extrinsic_value_estimates"] = buffer["environment_rewards"]
    buffer["curiosity_rewards"] = buffer["environment_rewards"]
    buffer["curiosity_returns"] = buffer["environment_rewards"]
    buffer["curiosity_value_estimates"] = buffer["environment_rewards"]
    buffer["advantages"] = buffer["environment_rewards"]
    trainer.trainer.update_buffer = buffer

    # when ghost trainer advance and wrapped trainer buffers full
    # the wrapped trainer pushes updated policy to correct queue
    assert policy_queue0.empty() and policy_queue1.empty()
    trainer.advance()
    assert not policy_queue0.empty() and policy_queue1.empty()


if __name__ == "__main__":
    pytest.main()
