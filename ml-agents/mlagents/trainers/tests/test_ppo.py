import unittest.mock as mock
import pytest

import numpy as np
import tensorflow as tf
import yaml

from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.ppo.trainer import discount_rewards
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.envs import UnityEnvironment
from mlagents.envs.mock_communicator import MockCommunicator


@pytest.fixture
def dummy_config():
    return yaml.load(
        '''
        trainer: ppo
        batch_size: 32
        beta: 5.0e-3
        buffer_size: 512
        epsilon: 0.2
        gamma: 0.99
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
        use_curiosity: false
        curiosity_strength: 0.0
        curiosity_enc_size: 1
        ''')


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_policy_evaluate(mock_communicator, mock_launcher, dummy_config):
    tf.reset_default_graph()
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    env = UnityEnvironment(' ')
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]

    trainer_parameters = dummy_config
    model_path = env.brain_names[0]
    trainer_parameters['model_path'] = model_path
    trainer_parameters['keep_checkpoints'] = 3
    policy = PPOPolicy(0, env.brains[env.brain_names[0]], trainer_parameters, False, False)
    run_out = policy.evaluate(brain_info)
    assert run_out['action'].shape == (3, 2)
    env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_cc_vector(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0)
            env = UnityEnvironment(' ')

            model = PPOModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.log_probs, model.value, model.entropy,
                        model.learning_rate]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]],),
                         model.epsilon: np.array([[0, 1], [2, 3]])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_cc_visual(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=2)
            env = UnityEnvironment(' ')

            model = PPOModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.log_probs, model.value, model.entropy,
                        model.learning_rate]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.visual_in[1]: np.ones([2, 40, 30, 3]),
                         model.epsilon: np.array([[0, 1], [2, 3]])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_dc_visual(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=2)
            env = UnityEnvironment(' ')
            model = PPOModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.visual_in[1]: np.ones([2, 40, 30, 3]),
                         model.action_masks: np.ones([2, 2],)
                         }
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_dc_vector(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0)
            env = UnityEnvironment(' ')
            model = PPOModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.action_masks: np.ones([2, 2])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_dc_vector_rnn(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0)
            env = UnityEnvironment(' ')
            memory_size = 128
            model = PPOModel(env.brains["RealFakeBrain"], use_recurrent=True, m_size=memory_size)
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate, model.memory_out]
            feed_dict = {model.batch_size: 1,
                         model.sequence_length: 2,
                         model.prev_action: [[0], [0]],
                         model.memory_in: np.zeros((1, memory_size)),
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.action_masks: np.ones([1, 2])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_cc_vector_rnn(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0)
            env = UnityEnvironment(' ')
            memory_size = 128
            model = PPOModel(env.brains["RealFakeBrain"], use_recurrent=True, m_size=memory_size)
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate, model.memory_out]
            feed_dict = {model.batch_size: 1,
                         model.sequence_length: 2,
                         model.memory_in: np.zeros((1, memory_size)),
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.epsilon: np.array([[0, 1]])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_dc_vector_curio(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0)
            env = UnityEnvironment(' ')
            model = PPOModel(env.brains["RealFakeBrain"], use_curiosity=True)
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate, model.intrinsic_reward]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.next_vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                         [3, 4, 5, 3, 4, 5]]),
                         model.action_holder: [[0], [0]],
                         model.action_masks: np.ones([2,2])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_cc_vector_curio(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0)
            env = UnityEnvironment(' ')
            model = PPOModel(env.brains["RealFakeBrain"], use_curiosity=True)
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate, model.intrinsic_reward]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.next_vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                         [3, 4, 5, 3, 4, 5]]),
                         model.output: [[0.0, 0.0], [0.0, 0.0]],
                         model.epsilon: np.array([[0, 1], [2, 3]])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_dc_visual_curio(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=2)
            env = UnityEnvironment(' ')
            model = PPOModel(env.brains["RealFakeBrain"], use_curiosity=True)
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate, model.intrinsic_reward]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.next_vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                         [3, 4, 5, 3, 4, 5]]),
                         model.action_holder: [[0], [0]],
                         model.visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.visual_in[1]: np.ones([2, 40, 30, 3]),
                         model.next_visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.next_visual_in[1]: np.ones([2, 40, 30, 3]),
                         model.action_masks: np.ones([2, 2])
                         }
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_ppo_model_cc_visual_curio(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=2)
            env = UnityEnvironment(' ')
            model = PPOModel(env.brains["RealFakeBrain"], use_curiosity=True)
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_log_probs, model.value, model.entropy,
                        model.learning_rate, model.intrinsic_reward]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                    [3, 4, 5, 3, 4, 5]]),
                         model.next_vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                         [3, 4, 5, 3, 4, 5]]),
                         model.output: [[0.0, 0.0], [0.0, 0.0]],
                         model.visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.visual_in[1]: np.ones([2, 40, 30, 3]),
                         model.next_visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.next_visual_in[1]: np.ones([2, 40, 30, 3]),
                         model.epsilon: np.array([[0, 1], [2, 3]])
                         }
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


def test_rl_functions():
    rewards = np.array([0.0, 0.0, 0.0, 1.0])
    gamma = 0.9
    returns = discount_rewards(rewards, gamma, 0.0)
    np.testing.assert_array_almost_equal(returns, np.array([0.729, 0.81, 0.9, 1.0]))


if __name__ == '__main__':
    pytest.main()
