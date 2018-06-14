import unittest.mock as mock
import pytest

import numpy as np
import tensorflow as tf

from unitytrainers.ppo.models import PPOModel
from unityagents import UnityEnvironment
from .mock_communicator import MockCommunicator


@mock.patch('unityagents.UnityEnvironment.executable_launcher')
@mock.patch('unityagents.UnityEnvironment.get_communicator')
def test_ppo_model_continuous(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete=False, visual_inputs=0)
            env = UnityEnvironment(' ')

            model = PPOModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.probs, model.value, model.entropy,
                        model.learning_rate]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                   [3, 4, 5, 3, 4, 5]])}
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch('unityagents.UnityEnvironment.executable_launcher')
@mock.patch('unityagents.UnityEnvironment.get_communicator')
def test_ppo_model_discrete(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete=True, visual_inputs=2)
            env = UnityEnvironment(' ')
            model = PPOModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.all_probs, model.value, model.entropy,
                        model.learning_rate]
            feed_dict = {model.batch_size: 2,
                         model.sequence_length: 1,
                         model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                   [3, 4, 5, 3, 4, 5]]),
                         model.visual_in[0]: np.ones([2, 40, 30, 3]),
                         model.visual_in[1]: np.ones([2, 40, 30, 3])
                         }
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


if __name__ == '__main__':
    pytest.main()
