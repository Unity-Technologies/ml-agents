from unittest import mock
import pytest

import numpy as np
from mlagents.tf_utils import tf
import copy
import attr
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers

from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.ppo.trainer import PPOTrainer, discount_rewards
from mlagents.trainers.ppo.optimizer_tf import PPOOptimizer
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.tests.test_simple_rl import PPO_CONFIG
from mlagents.trainers.tests.test_reward_signals import (  # noqa: F401; pylint: disable=unused-variable
    curiosity_dummy_config,
    gail_dummy_config,
)


@pytest.fixture
def dummy_config():
    return copy.deepcopy(PPO_CONFIG)


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 12


def _create_ppo_optimizer_ops_mock(dummy_config, use_rnn, use_discrete, use_visual):
    mock_specs = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )

    trainer_settings = attr.evolve(dummy_config)
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=16, memory_size=10)
        if use_rnn
        else None
    )
    policy = TFPolicy(
        0, mock_specs, trainer_settings, "test", False, create_tf_graph=False
    )
    optimizer = PPOOptimizer(policy, trainer_settings)
    policy.initialize()
    return optimizer


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_ppo_optimizer_update(dummy_config, rnn, visual, discrete):
    # Test evaluate
    tf.reset_default_graph()
    optimizer = _create_ppo_optimizer_ops_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec
    )
    # Mock out reward signal eval
    update_buffer["advantages"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_returns"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_value_estimates"] = update_buffer["environment_rewards"]
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
# We need to test this separately from test_reward_signals.py to ensure no interactions
def test_ppo_optimizer_update_curiosity(
    dummy_config, curiosity_dummy_config, rnn, visual, discrete  # noqa: F811
):
    # Test evaluate
    tf.reset_default_graph()
    dummy_config.reward_signals = curiosity_dummy_config
    optimizer = _create_ppo_optimizer_ops_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec
    )
    # Mock out reward signal eval
    update_buffer["advantages"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_returns"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_value_estimates"] = update_buffer["environment_rewards"]
    update_buffer["curiosity_returns"] = update_buffer["environment_rewards"]
    update_buffer["curiosity_value_estimates"] = update_buffer["environment_rewards"]
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


# We need to test this separately from test_reward_signals.py to ensure no interactions
def test_ppo_optimizer_update_gail(gail_dummy_config, dummy_config):  # noqa: F811
    # Test evaluate
    tf.reset_default_graph()
    dummy_config.reward_signals = gail_dummy_config
    optimizer = _create_ppo_optimizer_ops_mock(
        PPO_CONFIG, use_rnn=False, use_discrete=False, use_visual=False
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec
    )
    # Mock out reward signal eval
    update_buffer["advantages"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_returns"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_value_estimates"] = update_buffer["environment_rewards"]
    update_buffer["gail_returns"] = update_buffer["environment_rewards"]
    update_buffer["gail_value_estimates"] = update_buffer["environment_rewards"]
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )

    # Check if buffer size is too big
    update_buffer = mb.simulate_rollout(3000, optimizer.policy.behavior_spec)
    # Mock out reward signal eval
    update_buffer["advantages"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_returns"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_value_estimates"] = update_buffer["environment_rewards"]
    update_buffer["gail_returns"] = update_buffer["environment_rewards"]
    update_buffer["gail_value_estimates"] = update_buffer["environment_rewards"]
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_ppo_get_value_estimates(dummy_config, rnn, visual, discrete):
    tf.reset_default_graph()

    optimizer = _create_ppo_optimizer_ops_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        observation_shapes=optimizer.policy.behavior_spec.observation_shapes,
        max_step_complete=True,
        action_space=DISCRETE_ACTION_SPACE if discrete else VECTOR_ACTION_SPACE,
        is_discrete=discrete,
    )
    run_out, final_value_out = optimizer.get_trajectory_value_estimates(
        trajectory.to_agentbuffer(), trajectory.next_obs, done=False
    )
    for key, val in run_out.items():
        assert type(key) is str
        assert len(val) == 15

    run_out, final_value_out = optimizer.get_trajectory_value_estimates(
        trajectory.to_agentbuffer(), trajectory.next_obs, done=True
    )
    for key, val in final_value_out.items():
        assert type(key) is str
        assert val == 0.0

    # Check if we ignore terminal states properly
    optimizer.reward_signals["extrinsic"].use_terminal_states = False
    run_out, final_value_out = optimizer.get_trajectory_value_estimates(
        trajectory.to_agentbuffer(), trajectory.next_obs, done=False
    )
    for key, val in final_value_out.items():
        assert type(key) is str
        assert val != 0.0


def test_rl_functions():
    rewards = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    gamma = 0.9
    returns = discount_rewards(rewards, gamma, 0.0)
    np.testing.assert_array_almost_equal(
        returns, np.array([0.729, 0.81, 0.9, 1.0], dtype=np.float32)
    )


@mock.patch.object(RLTrainer, "create_model_saver")
@mock.patch("mlagents.trainers.ppo.trainer.PPOOptimizer")
def test_trainer_increment_step(ppo_optimizer, mock_create_model_saver):
    trainer_params = PPO_CONFIG
    mock_optimizer = mock.Mock()
    mock_optimizer.reward_signals = {}
    ppo_optimizer.return_value = mock_optimizer

    trainer = PPOTrainer("test_brain", 0, trainer_params, True, False, 0, "0")
    policy_mock = mock.Mock(spec=TFPolicy)
    policy_mock.get_current_step.return_value = 0
    step_count = (
        5  # 10 hacked because this function is no longer called through trainer
    )
    policy_mock.increment_step = mock.Mock(return_value=step_count)
    behavior_id = BehaviorIdentifiers.from_name_behavior_id(trainer.brain_name)
    trainer.add_policy(behavior_id, policy_mock)

    trainer._increment_step(5, trainer.brain_name)
    policy_mock.increment_step.assert_called_with(5)
    assert trainer.step == step_count


@pytest.mark.parametrize("use_discrete", [True, False])
def test_trainer_update_policy(
    dummy_config, curiosity_dummy_config, use_discrete  # noqa: F811
):
    mock_behavior_spec = mb.setup_test_behavior_specs(
        use_discrete,
        False,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )

    trainer_params = dummy_config
    trainer_params.network_settings.memory = NetworkSettings.MemorySettings(
        memory_size=10, sequence_length=16
    )

    # Test curiosity reward signal
    trainer_params.reward_signals = curiosity_dummy_config
    mock_brain_name = "MockBrain"
    behavior_id = BehaviorIdentifiers.from_name_behavior_id(mock_brain_name)
    trainer = PPOTrainer("test", 0, trainer_params, True, False, 0, "0")
    policy = trainer.create_policy(behavior_id, mock_behavior_spec)
    trainer.add_policy(behavior_id, policy)
    # Test update with sequence length smaller than batch size
    buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, mock_behavior_spec)
    # Mock out reward signal eval
    buffer["extrinsic_rewards"] = buffer["environment_rewards"]
    buffer["extrinsic_returns"] = buffer["environment_rewards"]
    buffer["extrinsic_value_estimates"] = buffer["environment_rewards"]
    buffer["curiosity_rewards"] = buffer["environment_rewards"]
    buffer["curiosity_returns"] = buffer["environment_rewards"]
    buffer["curiosity_value_estimates"] = buffer["environment_rewards"]
    buffer["advantages"] = buffer["environment_rewards"]

    trainer.update_buffer = buffer
    trainer._update_policy()


def test_process_trajectory(dummy_config):
    behavior_spec = mb.setup_test_behavior_specs(
        True,
        False,
        vector_action_space=DISCRETE_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )
    mock_brain_name = "MockBrain"
    behavior_id = BehaviorIdentifiers.from_name_behavior_id(mock_brain_name)
    trainer = PPOTrainer("test_brain", 0, dummy_config, True, False, 0, "0")
    policy = trainer.create_policy(behavior_id, behavior_spec)
    trainer.add_policy(behavior_id, policy)
    trajectory_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        observation_shapes=behavior_spec.observation_shapes,
        max_step_complete=True,
        action_space=[2],
    )
    trajectory_queue.put(trajectory)
    trainer.advance()

    # Check that trainer put trajectory in update buffer
    assert trainer.update_buffer.num_experiences == 15

    # Check that GAE worked
    assert (
        "advantages" in trainer.update_buffer
        and "discounted_returns" in trainer.update_buffer
    )

    # Check that the stats are being collected as episode isn't complete
    for reward in trainer.collected_rewards.values():
        for agent in reward.values():
            assert agent > 0

    # Add a terminal trajectory
    trajectory = make_fake_trajectory(
        length=time_horizon + 1,
        max_step_complete=False,
        observation_shapes=behavior_spec.observation_shapes,
        action_space=[2],
    )
    trajectory_queue.put(trajectory)
    trainer.advance()

    # Check that the stats are reset as episode is finished
    for reward in trainer.collected_rewards.values():
        for agent in reward.values():
            assert agent == 0
    assert trainer.stats_reporter.get_stats_summaries("Policy/Extrinsic Reward").num > 0


@mock.patch.object(RLTrainer, "create_model_saver")
@mock.patch("mlagents.trainers.ppo.trainer.PPOOptimizer")
def test_add_get_policy(ppo_optimizer, mock_create_model_saver, dummy_config):
    mock_optimizer = mock.Mock()
    mock_optimizer.reward_signals = {}
    ppo_optimizer.return_value = mock_optimizer

    trainer = PPOTrainer("test_policy", 0, dummy_config, True, False, 0, "0")
    policy = mock.Mock(spec=TFPolicy)
    policy.get_current_step.return_value = 2000

    behavior_id = BehaviorIdentifiers.from_name_behavior_id(trainer.brain_name)
    trainer.add_policy(behavior_id, policy)
    assert trainer.get_policy("test_policy") == policy

    # Make sure the summary steps were loaded properly
    assert trainer.get_step == 2000


if __name__ == "__main__":
    pytest.main()
