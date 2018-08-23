import unittest.mock as mock
import pytest

import numpy as np
import tensorflow as tf

from mlagents.trainers.bc.models import BehavioralCloningModel
from mlagents.envs import UnityEnvironment
from tests.mock_communicator import MockCommunicator


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_cc_bc_model(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0)
            env = UnityEnvironment(' ')
            model = BehavioralCloningModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.policy]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                   [3, 4, 5, 3, 4, 5]])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_dc_bc_model(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0)
            env = UnityEnvironment(' ')
            model = BehavioralCloningModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.action_probs]
            feed_dict = {model.batch_size: 2,
                         model.dropout_rate: 1.0,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                   [3, 4, 5, 3, 4, 5]]),
                         model.action_masks: np.ones([2,2])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_visual_dc_bc_model(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=2)
            env = UnityEnvironment(' ')
            model = BehavioralCloningModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.action_probs]
            feed_dict = {model.batch_size: 2,
                         model.dropout_rate: 1.0,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                   [3, 4, 5, 3, 4, 5]]),
                         model.visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.visual_in[1]: np.ones([2, 40, 30, 3]),
                         model.action_masks: np.ones([2,2])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_visual_cc_bc_model(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=2)
            env = UnityEnvironment(' ')
            model = BehavioralCloningModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.policy]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                   [3, 4, 5, 3, 4, 5]]),
                         model.visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.visual_in[1]: np.ones([2, 40, 30, 3])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


if __name__ == '__main__':
    pytest.main()
