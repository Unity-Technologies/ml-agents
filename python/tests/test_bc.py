import unittest.mock as mock
import pytest

import numpy as np
import tensorflow as tf

from unitytrainers.bc.models import BehavioralCloningModel
from unityagents import UnityEnvironment


def test_cc_bc_model():
    c_action_c_state_start = '''{
      "AcademyName": "RealFakeAcademy",
      "resetParameters": {},
      "brainNames": ["RealFakeBrain"],
      "externalBrainNames": ["RealFakeBrain"],
      "logPath":"RealFakePath",
      "apiNumber":"API-3",
      "brainParameters": [{
          "vectorObservationSize": 3,
          "numStackedVectorObservations": 2,
          "vectorActionSize": 2,
          "memorySize": 0,
          "cameraResolutions": [],
          "vectorActionDescriptions": ["",""],
          "vectorActionSpaceType": 1,
          "vectorObservationSpaceType": 1
          }]
    }'''.encode()

    tf.reset_default_graph()
    with mock.patch('subprocess.Popen'):
        with mock.patch('socket.socket') as mock_socket:
            with mock.patch('glob.glob') as mock_glob:
                # End of mock
                with tf.Session() as sess:
                    with tf.variable_scope("FakeGraphScope"):
                        mock_glob.return_value = ['FakeLaunchPath']
                        mock_socket.return_value.accept.return_value = (mock_socket, 0)
                        mock_socket.recv.return_value.decode.return_value = c_action_c_state_start
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


def test_dc_bc_model():
    d_action_c_state_start = '''{
      "AcademyName": "RealFakeAcademy",
      "resetParameters": {},
      "brainNames": ["RealFakeBrain"],
      "externalBrainNames": ["RealFakeBrain"],
      "logPath":"RealFakePath",
      "apiNumber":"API-3",
      "brainParameters": [{
          "vectorObservationSize": 3,
          "numStackedVectorObservations": 2,
          "vectorActionSize": 2,
          "memorySize": 0,
          "cameraResolutions": [{"width":30,"height":40,"blackAndWhite":false}],
          "vectorActionDescriptions": ["",""],
          "vectorActionSpaceType": 0,
          "vectorObservationSpaceType": 1
          }]
    }'''.encode()

    tf.reset_default_graph()
    with mock.patch('subprocess.Popen'):
        with mock.patch('socket.socket') as mock_socket:
            with mock.patch('glob.glob') as mock_glob:
                with tf.Session() as sess:
                    with tf.variable_scope("FakeGraphScope"):
                        mock_glob.return_value = ['FakeLaunchPath']
                        mock_socket.return_value.accept.return_value = (mock_socket, 0)
                        mock_socket.recv.return_value.decode.return_value = d_action_c_state_start
                        env = UnityEnvironment(' ')

                        model = BehavioralCloningModel(env.brains["RealFakeBrain"])
                        init = tf.global_variables_initializer()
                        sess.run(init)

                        run_list = [model.sample_action, model.policy]
                        feed_dict = {model.batch_size: 2,
                                     model.dropout_rate: 1.0,
                                     model.sequence_length: 1,
                                     model.vector_in: np.array([[1, 2, 3, 1, 2, 3],
                                                               [3, 4, 5, 3, 4, 5]]),
                                     model.visual_in[0]: np.ones([2, 40, 30, 3])}
                        sess.run(run_list, feed_dict=feed_dict)
                        env.close()


if __name__ == '__main__':
    pytest.main()
