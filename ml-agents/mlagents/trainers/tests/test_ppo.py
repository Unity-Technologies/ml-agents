from unittest import mock
import pytest

import numpy as np
from mlagents.tf_utils import tf

import yaml

from mlagents.trainers.ppo.trainer import PPOTrainer, discount_rewards
from mlagents.trainers.ppo.optimizer import PPOOptimizer
from mlagents.trainers.policy.nn_policy import NNPolicy
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import make_brain_parameters
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.tests.test_reward_signals import (  # noqa: F401; pylint: disable=unused-variable
    curiosity_dummy_config,
    gail_dummy_config,
)


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
        sequence_length: 16
        summary_freq: 1000
        use_recurrent: false
        normalize: true
        memory_size: 10
        curiosity_strength: 0.0
        curiosity_enc_size: 1
        summary_path: test
        model_path: test
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


VECTOR_ACTION_SPACE = [2]
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 12


def _create_ppo_optimizer_ops_mock(dummy_config, use_rnn, use_discrete, use_visual):
    mock_brain = mb.setup_mock_brain(
        use_discrete,
        use_visual,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_parameters = dummy_config
    model_path = "testmodel"
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["use_recurrent"] = use_rnn
    policy = NNPolicy(
        0, mock_brain, trainer_parameters, False, False, create_tf_graph=False
    )
    optimizer = PPOOptimizer(policy, trainer_parameters)
    return optimizer


def _create_fake_trajectory(use_discrete, use_visual, time_horizon):
    if use_discrete:
        act_space = DISCRETE_ACTION_SPACE
    else:
        act_space = VECTOR_ACTION_SPACE

    if use_visual:
        num_vis_obs = 1
        vec_obs_size = 0
    else:
        num_vis_obs = 0
        vec_obs_size = VECTOR_OBS_SPACE

    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        vec_obs_size=vec_obs_size,
        num_vis_obs=num_vis_obs,
        action_space=act_space,
    )
    return trajectory


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
    update_buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, optimizer.policy.brain)
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
    curiosity_dummy_config, dummy_config, rnn, visual, discrete  # noqa: F811
):
    # Test evaluate
    tf.reset_default_graph()
    dummy_config["reward_signals"].update(curiosity_dummy_config)
    optimizer = _create_ppo_optimizer_ops_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, optimizer.policy.brain)
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
    dummy_config["reward_signals"].update(gail_dummy_config)
    optimizer = _create_ppo_optimizer_ops_mock(
        dummy_config, use_rnn=False, use_discrete=False, use_visual=False
    )
    # Test update
    update_buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, optimizer.policy.brain)
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
    update_buffer = mb.simulate_rollout(3000, optimizer.policy.brain)
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
    trajectory = _create_fake_trajectory(discrete, visual, time_horizon)
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


@mock.patch("mlagents.trainers.ppo.trainer.PPOOptimizer")
def test_trainer_increment_step(ppo_optimizer, dummy_config):
    trainer_params = dummy_config
    mock_optimizer = mock.Mock()
    mock_optimizer.reward_signals = {}
    ppo_optimizer.return_value = mock_optimizer

    brain_params = BrainParameters(
        brain_name="test_brain",
        vector_observation_space_size=1,
        camera_resolutions=[],
        vector_action_space_size=[2],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )

    trainer = PPOTrainer(
        brain_params.brain_name, 0, trainer_params, True, False, 0, "0"
    )
    policy_mock = mock.Mock(spec=NNPolicy)
    policy_mock.get_current_step.return_value = 0
    step_count = (
        5
    )  # 10 hacked because this function is no longer called through trainer
    policy_mock.increment_step = mock.Mock(return_value=step_count)
    trainer.add_policy("testbehavior", policy_mock)

    trainer._increment_step(5, "testbehavior")
    policy_mock.increment_step.assert_called_with(5)
    assert trainer.step == step_count


@pytest.mark.parametrize("use_discrete", [True, False])
def test_trainer_update_policy(dummy_config, use_discrete):
    mock_brain = mb.setup_mock_brain(
        use_discrete,
        False,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_params = dummy_config
    trainer_params["use_recurrent"] = True

    # Test curiosity reward signal
    trainer_params["reward_signals"]["curiosity"] = {}
    trainer_params["reward_signals"]["curiosity"]["strength"] = 1.0
    trainer_params["reward_signals"]["curiosity"]["gamma"] = 0.99
    trainer_params["reward_signals"]["curiosity"]["encoding_size"] = 128

    trainer = PPOTrainer(mock_brain.brain_name, 0, trainer_params, True, False, 0, "0")
    policy = trainer.create_policy(mock_brain)
    trainer.add_policy(mock_brain.brain_name, policy)
    # Test update with sequence length smaller than batch size
    buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, mock_brain)
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
    brain_params = BrainParameters(
        brain_name="test_brain",
        vector_observation_space_size=1,
        camera_resolutions=[],
        vector_action_space_size=[2],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )
    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    trainer = PPOTrainer(brain_params, 0, dummy_config, True, False, 0, "0")
    policy = trainer.create_policy(brain_params)
    trainer.add_policy(brain_params.brain_name, policy)
    trajectory_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    time_horizon = 15
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        vec_obs_size=1,
        num_vis_obs=0,
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
        vec_obs_size=1,
        num_vis_obs=0,
        action_space=[2],
    )
    trajectory_queue.put(trajectory)
    trainer.advance()

    # Check that the stats are reset as episode is finished
    for reward in trainer.collected_rewards.values():
        for agent in reward.values():
            assert agent == 0
    assert trainer.stats_reporter.get_stats_summaries("Policy/Extrinsic Reward").num > 0


@mock.patch("mlagents.trainers.ppo.trainer.PPOOptimizer")
def test_add_get_policy(ppo_optimizer, dummy_config):
    brain_params = make_brain_parameters(
        discrete_action=False, visual_inputs=0, vec_obs_size=6
    )
    mock_optimizer = mock.Mock()
    mock_optimizer.reward_signals = {}
    ppo_optimizer.return_value = mock_optimizer

    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    trainer = PPOTrainer(brain_params, 0, dummy_config, True, False, 0, "0")
    policy = mock.Mock(spec=NNPolicy)
    policy.get_current_step.return_value = 2000

    trainer.add_policy(brain_params.brain_name, policy)
    assert trainer.get_policy(brain_params.brain_name) == policy

    # Make sure the summary steps were loaded properly
    assert trainer.get_step == 2000
    assert trainer.next_summary_step > 2000

    # Test incorrect class of policy
    policy = mock.Mock()
    with pytest.raises(RuntimeError):
        trainer.add_policy(brain_params, policy)


def test_bad_config(dummy_config):
    brain_params = make_brain_parameters(
        discrete_action=False, visual_inputs=0, vec_obs_size=6
    )
    # Test that we throw an error if we have sequence length greater than batch size
    dummy_config["sequence_length"] = 64
    dummy_config["batch_size"] = 32
    dummy_config["use_recurrent"] = True
    with pytest.raises(UnityTrainerException):
        _ = PPOTrainer(brain_params, 0, dummy_config, True, False, 0, "0")


if __name__ == "__main__":
    pytest.main()
