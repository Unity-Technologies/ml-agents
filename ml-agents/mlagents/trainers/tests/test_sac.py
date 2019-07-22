import unittest.mock as mock
import pytest
import tempfile
import yaml
import math

import numpy as np
import tensorflow as tf

from mlagents.trainers.sac.models import SACModel
from mlagents.trainers.sac.policy import SACPolicy
from mlagents.trainers.tests.test_environments.test_simple import (
    Simple1DEnvironment,
    SimpleEnvManager,
)
from mlagents.envs import UnityEnvironment
from mlagents.envs.mock_communicator import MockCommunicator
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs import BrainInfo, AllBrainInfo, BrainParameters
from mlagents.envs.communicator_objects import AgentInfoProto
from mlagents.envs.simple_env_manager import SimpleEnvManager


@pytest.fixture
def dummy_config():
    return yaml.load(
        """
        trainer: sac
        batch_size: 32
        buffer_size: 10240
        buffer_init_steps: 0
        hidden_units: 32
        init_entcoef: 0.1
        learning_rate: 3.0e-4
        max_steps: 1024
        memory_size: 256
        normalize: false
        updates_per_train: 1
        train_interval: 1
        num_layers: 1
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        tau: 0.005
        use_recurrent: false
        curiosity_enc_size: 128
        demo_path: None
        vis_encode_type: default
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
        """
    )


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_sac_policy_evaluate(mock_communicator, mock_launcher, dummy_config):
    tf.reset_default_graph()
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    brain_infos = env.reset()
    brain_info = brain_infos[env.brain_names[0]]

    trainer_parameters = dummy_config
    model_path = env.brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    policy = SACPolicy(
        0, env.brains[env.brain_names[0]], trainer_parameters, False, False
    )
    run_out = policy.evaluate(brain_info)
    assert run_out["action"].shape == (3, 2)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_sac_model_cc_vector(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0
            )
            env = UnityEnvironment(" ")

            model = SACModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
            }
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_sac_model_cc_visual(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=2
            )
            env = UnityEnvironment(" ")

            model = SACModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.visual_in[0]: np.ones([2, 40, 30, 3]),
                model.visual_in[1]: np.ones([2, 40, 30, 3]),
            }
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


@mock.patch("mlagents.envs.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.UnityEnvironment.get_communicator")
def test_sac_model_dc_visual(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=2
            )
            env = UnityEnvironment(" ")
            model = SACModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
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
def test_sac_model_dc_vector(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0
            )
            env = UnityEnvironment(" ")
            model = SACModel(env.brains["RealFakeBrain"])
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.output, model.value, model.entropy, model.learning_rate]
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
def test_sac_model_dc_vector_rnn(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=0
            )
            env = UnityEnvironment(" ")
            memory_size = 128
            model = SACModel(
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
def test_sac_model_cc_vector_rnn(mock_communicator, mock_launcher):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, visual_inputs=0
            )
            env = UnityEnvironment(" ")
            memory_size = 128
            model = SACModel(
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
            }
            sess.run(run_list, feed_dict=feed_dict)
            env.close()


def test_sac_simple_env(dummy_config):
    # Create controller and begin training.
    with tempfile.TemporaryDirectory() as dir:
        run_id = "id"
        save_freq = 99999
        tc = TrainerController(
            dir,
            dir,
            run_id,
            save_freq,
            meta_curriculum=None,
            load=False,
            train=True,
            keep_checkpoints=1,
            lesson=None,
            training_seed=1337,
            fast_simulation=True,
        )

        # Begin training
        env = Simple1DEnvironment()
        env_manager = SimpleEnvManager(env)
        tc.start_learning(env_manager, {"default": dummy_config})

        for _, mean_reward in tc._get_measure_vals().items():
            assert not math.isnan(mean_reward)
            assert mean_reward > 0.9


if __name__ == "__main__":
    pytest.main()
