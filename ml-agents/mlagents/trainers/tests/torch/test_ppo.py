import pytest

import numpy as np
from mlagents.tf_utils import tf
import attr

from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.settings import NetworkSettings, FrameworkType
from mlagents.trainers.tests.dummy_config import (  # noqa: F401; pylint: disable=unused-variable
    ppo_dummy_config,
    curiosity_dummy_config,
    gail_dummy_config,
)

from mlagents_envs.base_env import ActionSpec


@pytest.fixture
def dummy_config():
    return attr.evolve(ppo_dummy_config(), framework=FrameworkType.PYTORCH)


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 12

CONTINUOUS_ACTION_SPEC = ActionSpec.create_continuous(VECTOR_ACTION_SPACE)
DISCRETE_ACTION_SPEC = ActionSpec.create_discrete(tuple(DISCRETE_ACTION_SPACE))


def create_test_ppo_optimizer(dummy_config, use_rnn, use_discrete, use_visual):
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
    policy = TorchPolicy(0, mock_specs, trainer_settings, "test", False)
    optimizer = TorchPPOOptimizer(policy, trainer_settings)
    return optimizer


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_ppo_optimizer_update(dummy_config, rnn, visual, discrete):
    # Test evaluate
    tf.reset_default_graph()
    optimizer = create_test_ppo_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES,
        optimizer.policy.behavior_spec,
        memory_size=optimizer.policy.m_size,
    )
    # Mock out reward signal eval
    update_buffer["advantages"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_returns"] = update_buffer["environment_rewards"]
    update_buffer["extrinsic_value_estimates"] = update_buffer["environment_rewards"]

    return_stats = optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )
    # Make sure we have the right stats
    required_stats = [
        "Losses/Policy Loss",
        "Losses/Value Loss",
        "Policy/Learning Rate",
        "Policy/Epsilon",
        "Policy/Beta",
    ]
    for stat in required_stats:
        assert stat in return_stats.keys()


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
    optimizer = create_test_ppo_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES,
        optimizer.policy.behavior_spec,
        memory_size=optimizer.policy.m_size,
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
    dummy_config.reward_signals = gail_dummy_config
    config = attr.evolve(ppo_dummy_config(), framework=FrameworkType.PYTORCH)
    optimizer = create_test_ppo_optimizer(
        config, use_rnn=False, use_discrete=False, use_visual=False
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
    update_buffer["continuous_log_probs"] = np.ones_like(
        update_buffer["continuous_action"]
    )
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
    optimizer = create_test_ppo_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        observation_shapes=optimizer.policy.behavior_spec.observation_shapes,
        action_spec=DISCRETE_ACTION_SPEC if discrete else CONTINUOUS_ACTION_SPEC,
        max_step_complete=True,
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


if __name__ == "__main__":
    pytest.main()
