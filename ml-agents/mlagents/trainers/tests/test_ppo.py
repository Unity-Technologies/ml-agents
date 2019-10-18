import unittest.mock as mock
import pytest

import numpy as np
import tensorflow as tf
import yaml

from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.ppo.trainer import PPOTrainer, discount_rewards
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.rl_trainer import AllRewardsOutput
from mlagents.trainers.components.reward_signals import RewardSignalResult
from mlagents.envs.brain import BrainParameters
from mlagents.envs.environment import UnityEnvironment
from mlagents.envs.mock_communicator import MockCommunicator
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import make_brain_parameters


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
        memory_size: 8
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
BUFFER_INIT_SAMPLES = 32
NUM_AGENTS = 12


@mock.patch("mlagents.envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.environment.UnityEnvironment.get_communicator")
def test_ppo_policy_evaluate(mock_communicator, mock_launcher, dummy_config):
    tf.reset_default_graph()
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    brain_infos = env.reset()
    brain_info = brain_infos[env.external_brain_names[0]]

    trainer_parameters = dummy_config
    model_path = env.external_brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    policy = PPOPolicy(
        0, env.brains[env.external_brain_names[0]], trainer_parameters, False, False
    )
    run_out = policy.evaluate(brain_info)
    assert run_out["action"].shape == (3, 2)
    env.close()


@mock.patch("mlagents.envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.environment.UnityEnvironment.get_communicator")
def test_ppo_get_value_estimates(mock_communicator, mock_launcher, dummy_config):
    tf.reset_default_graph()
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    brain_infos = env.reset()
    brain_info = brain_infos[env.external_brain_names[0]]

    trainer_parameters = dummy_config
    model_path = env.external_brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    policy = PPOPolicy(
        0, env.brains[env.external_brain_names[0]], trainer_parameters, False, False
    )
    run_out = policy.get_value_estimates(brain_info, 0, done=False)
    for key, val in run_out.items():
        assert type(key) is str
        assert type(val) is float

    run_out = policy.get_value_estimates(brain_info, 0, done=True)
    for key, val in run_out.items():
        assert type(key) is str
        assert val == 0.0

    # Check if we ignore terminal states properly
    policy.reward_signals["extrinsic"].use_terminal_states = False
    run_out = policy.get_value_estimates(brain_info, 0, done=True)
    for key, val in run_out.items():
        assert type(key) is str
        assert val != 0.0

    env.close()


def test_ppo_model_cc_vector():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = PPOModel(
                make_brain_parameters(discrete_action=False, visual_inputs=0)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
            ]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.epsilon: np.array([[0, 1], [2, 3]]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_ppo_model_cc_visual():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = PPOModel(
                make_brain_parameters(discrete_action=False, visual_inputs=2)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
            ]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.visual_in[0]: np.ones([2, 40, 30, 3]),
                model.visual_in[1]: np.ones([2, 40, 30, 3]),
                model.epsilon: np.array([[0, 1], [2, 3]]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_ppo_model_dc_visual():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = PPOModel(
                make_brain_parameters(discrete_action=True, visual_inputs=2)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.all_log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
            ]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.visual_in[0]: np.ones([2, 40, 30, 3]),
                model.visual_in[1]: np.ones([2, 40, 30, 3]),
                model.action_masks: np.ones([2, 2]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_ppo_model_dc_vector():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = PPOModel(
                make_brain_parameters(discrete_action=True, visual_inputs=0)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.all_log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
            ]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.action_masks: np.ones([2, 2]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_ppo_model_dc_vector_rnn():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            memory_size = 128
            model = PPOModel(
                make_brain_parameters(discrete_action=True, visual_inputs=0),
                use_recurrent=True,
                m_size=memory_size,
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.all_log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
                model.memory_out,
            ]
            feed_dict = {
                model.batch_size: 1,
                model.sequence_length: 2,
                model.prev_action: [[0], [0]],
                model.memory_in: np.zeros((1, memory_size)),
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.action_masks: np.ones([1, 2]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_ppo_model_cc_vector_rnn():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            memory_size = 128
            model = PPOModel(
                make_brain_parameters(discrete_action=False, visual_inputs=0),
                use_recurrent=True,
                m_size=memory_size,
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [
                model.output,
                model.all_log_probs,
                model.value,
                model.entropy,
                model.learning_rate,
                model.memory_out,
            ]
            feed_dict = {
                model.batch_size: 1,
                model.sequence_length: 2,
                model.memory_in: np.zeros((1, memory_size)),
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.epsilon: np.array([[0, 1]]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_rl_functions():
    rewards = np.array([0.0, 0.0, 0.0, 1.0])
    gamma = 0.9
    returns = discount_rewards(rewards, gamma, 0.0)
    np.testing.assert_array_almost_equal(returns, np.array([0.729, 0.81, 0.9, 1.0]))


def test_trainer_increment_step(dummy_config):
    trainer_params = dummy_config
    brain_params = BrainParameters("test_brain", 1, 1, [], [2], [], 0)

    trainer = PPOTrainer(brain_params, 0, trainer_params, True, False, 0, "0", False)
    policy_mock = mock.Mock()
    step_count = 10
    policy_mock.increment_step = mock.Mock(return_value=step_count)
    trainer.policy = policy_mock

    trainer.increment_step(5)
    policy_mock.increment_step.assert_called_with(5)
    assert trainer.step == 10


@mock.patch("mlagents.envs.environment.UnityEnvironment")
@pytest.mark.parametrize("use_discrete", [True, False])
def test_trainer_update_policy(mock_env, dummy_config, use_discrete):
    env, mock_brain, _ = mb.setup_mock_env_and_brains(
        mock_env,
        use_discrete,
        False,
        num_agents=NUM_AGENTS,
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

    trainer = PPOTrainer(mock_brain, 0, trainer_params, True, False, 0, "0", False)
    # Test update with sequence length smaller than batch size
    buffer = mb.simulate_rollout(env, trainer.policy, BUFFER_INIT_SAMPLES)
    # Mock out reward signal eval
    buffer.update_buffer["extrinsic_rewards"] = buffer.update_buffer["rewards"]
    buffer.update_buffer["extrinsic_returns"] = buffer.update_buffer["rewards"]
    buffer.update_buffer["extrinsic_value_estimates"] = buffer.update_buffer["rewards"]
    buffer.update_buffer["curiosity_rewards"] = buffer.update_buffer["rewards"]
    buffer.update_buffer["curiosity_returns"] = buffer.update_buffer["rewards"]
    buffer.update_buffer["curiosity_value_estimates"] = buffer.update_buffer["rewards"]

    trainer.training_buffer = buffer
    trainer.update_policy()
    # Make batch length a larger multiple of sequence length
    trainer.trainer_parameters["batch_size"] = 128
    trainer.update_policy()
    # Make batch length a larger non-multiple of sequence length
    trainer.trainer_parameters["batch_size"] = 100
    trainer.update_policy()


def test_add_rewards_output(dummy_config):
    brain_params = BrainParameters("test_brain", 1, 1, [], [2], [], 0)
    dummy_config["summary_path"] = "./summaries/test_trainer_summary"
    dummy_config["model_path"] = "./models/test_trainer_models/TestModel"
    trainer = PPOTrainer(brain_params, 0, dummy_config, True, False, 0, "0", False)
    rewardsout = AllRewardsOutput(
        reward_signals={
            "extrinsic": RewardSignalResult(
                scaled_reward=np.array([1.0, 1.0]), unscaled_reward=np.array([1.0, 1.0])
            )
        },
        environment=np.array([1.0, 1.0]),
    )
    values = {"extrinsic": np.array([[2.0]])}
    agent_id = "123"
    idx = 0
    # make sure that we're grabbing from the next_idx for rewards. If we're not, the test will fail.
    next_idx = 1
    trainer.add_rewards_outputs(
        rewardsout,
        values=values,
        agent_id=agent_id,
        agent_idx=idx,
        agent_next_idx=next_idx,
    )
    assert trainer.training_buffer[agent_id]["extrinsic_value_estimates"][0] == 2.0
    assert trainer.training_buffer[agent_id]["extrinsic_rewards"][0] == 1.0


if __name__ == "__main__":
    pytest.main()
