import unittest.mock as mock
import pytest

import numpy as np
import tensorflow as tf
import yaml
import os

from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.ppo.trainer import discount_rewards
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.envs import UnityEnvironment
from mlagents.envs.mock_communicator import MockCommunicator


@pytest.fixture
def dummy_config():
    return yaml.load(
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
        reward_signals:
          - extrinsic
        reward_strengths:
          - 1.0
        gammas:
          - 0.99
        """
    )


def create_ppo_policy(
    mock_communicator,
    mock_launcher,
    dummy_config,
    reward_signal,
    use_rnn,
    use_discrete,
    use_visual,
):
    tf.reset_default_graph()
    mock_communicator.return_value = MockCommunicator(
        discrete_action=use_discrete, visual_inputs=int(use_visual)
    )
    env = UnityEnvironment(" ")

    trainer_parameters = dummy_config
    model_path = env.brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["reward_signals"].append(reward_signal)
    trainer_parameters["reward_strengths"].append(0.1)
    trainer_parameters["gammas"].append(0.9)
    trainer_parameters["use_recurrent"] = use_rnn
    trainer_parameters["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/test.demo"
    )
    policy = PPOPolicy(
        0, env.brains[env.brain_names[0]], trainer_parameters, False, False
    )
    return env, policy


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_gail_cc_evaluate(mock_communicator, mock_launcher, dummy_config):
    env, policy = create_ppo_policy(
        mock_communicator, mock_launcher, dummy_config, "gail", False, False, False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]
    next_brain_info = env.step(6 * [0])[env.brain_names[0]]
    scaled_reward, unscaled_reward = policy.reward_signals["gail"].evaluate(
        brain_info, next_brain_info
    )
    assert scaled_reward.shape == (3,)
    assert unscaled_reward.shape == (3,)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_gail_dc_evaluate(mock_communicator, mock_launcher, dummy_config):
    env, policy = create_ppo_policy(
        mock_communicator, mock_launcher, dummy_config, "gail", False, True, False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]
    next_brain_info = env.step(3 * [0])[env.brain_names[0]]
    scaled_reward, unscaled_reward = policy.reward_signals["gail"].evaluate(
        brain_info, next_brain_info
    )
    assert scaled_reward.shape == (3,)
    assert unscaled_reward.shape == (3,)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_gail_rnn_evaluate(mock_communicator, mock_launcher, dummy_config):
    env, policy = create_ppo_policy(
        mock_communicator, mock_launcher, dummy_config, "gail", True, False, False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]
    next_brain_info = env.step(6 * [0])[env.brain_names[0]]
    scaled_reward, unscaled_reward = policy.reward_signals["gail"].evaluate(
        brain_info, next_brain_info
    )
    assert scaled_reward.shape == (3,)
    assert unscaled_reward.shape == (3,)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_curiosity_cc_evaluate(mock_communicator, mock_launcher, dummy_config):
    env, policy = create_ppo_policy(
        mock_communicator, mock_launcher, dummy_config, "curiosity", False, False, False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]
    next_brain_info = env.step(6 * [0])[env.brain_names[0]]
    scaled_reward, unscaled_reward = policy.reward_signals["curiosity"].evaluate(
        brain_info, next_brain_info
    )
    assert scaled_reward.shape == (3,)
    assert unscaled_reward.shape == (3,)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_curiosity_dc_evaluate(mock_communicator, mock_launcher, dummy_config):
    env, policy = create_ppo_policy(
        mock_communicator, mock_launcher, dummy_config, "curiosity", False, True, False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]
    next_brain_info = env.step(3 * [0])[env.brain_names[0]]
    scaled_reward, unscaled_reward = policy.reward_signals["curiosity"].evaluate(
        brain_info, next_brain_info
    )
    assert scaled_reward.shape == (3,)
    assert unscaled_reward.shape == (3,)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_curiosity_rnn_evaluate(mock_communicator, mock_launcher, dummy_config):
    env, policy = create_ppo_policy(
        mock_communicator, mock_launcher, dummy_config, "curiosity", True, False, False
    )
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]
    next_brain_info = env.step(6 * [0])[env.brain_names[0]]
    scaled_reward, unscaled_reward = policy.reward_signals["curiosity"].evaluate(
        brain_info, next_brain_info
    )
    assert scaled_reward.shape == (3,)
    assert unscaled_reward.shape == (3,)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_ppo_model_cc_vector(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0
            )
            env = UnityEnvironment(" ")

            model = PPOModel(env.brains["RealFakeBrain"])
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
            env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_ppo_model_cc_visual(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=2
            )
            env = UnityEnvironment(" ")

            model = PPOModel(env.brains["RealFakeBrain"])
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
            env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_ppo_model_dc_visual(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=2
            )
            env = UnityEnvironment(" ")
            model = PPOModel(env.brains["RealFakeBrain"])
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
            env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_ppo_model_dc_vector(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0
            )
            env = UnityEnvironment(" ")
            model = PPOModel(env.brains["RealFakeBrain"])
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
            env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_ppo_model_dc_vector_rnn(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0
            )
            env = UnityEnvironment(" ")
            memory_size = 128
            model = PPOModel(
                env.brains["RealFakeBrain"], use_recurrent=True, m_size=memory_size
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
            env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_ppo_model_cc_vector_rnn(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0
            )
            env = UnityEnvironment(" ")
            memory_size = 128
            model = PPOModel(
                env.brains["RealFakeBrain"], use_recurrent=True, m_size=memory_size
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
            env.close()


if __name__ == "__main__":
    pytest.main()
