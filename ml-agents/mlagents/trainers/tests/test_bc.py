import unittest.mock as mock
import pytest
import os

import numpy as np
import tensorflow as tf
import yaml

from mlagents.trainers.bc.models import BehavioralCloningModel
import mlagents.trainers.tests.mock_brain as mb
from mlagents.trainers.bc.policy import BCPolicy
from mlagents.trainers.bc.offline_trainer import BCTrainer
from mlagents.envs.environment import UnityEnvironment
from mlagents.envs.mock_communicator import MockCommunicator
from mlagents.trainers.tests.mock_brain import make_brain_parameters


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
            hidden_units: 32
            learning_rate: 3.0e-4
            num_layers: 1
            use_recurrent: false
            sequence_length: 32
            memory_size: 32
            batches_per_epoch: 1
            batch_size: 32
            summary_freq: 2000
            max_steps: 4000
            """
    )


def create_bc_trainer(dummy_config, is_discrete=False):
    mock_env = mock.Mock()
    if is_discrete:
        mock_brain = mb.create_mock_pushblock_brain()
        mock_braininfo = mb.create_mock_braininfo(
            num_agents=12, num_vector_observations=70
        )
    else:
        mock_brain = mb.create_mock_3dball_brain()
        mock_braininfo = mb.create_mock_braininfo(
            num_agents=12, num_vector_observations=8
        )
    mb.setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)
    env = mock_env()

    trainer_parameters = dummy_config
    trainer_parameters["summary_path"] = "tmp"
    trainer_parameters["model_path"] = "tmp"
    trainer_parameters["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/test.demo"
    )
    trainer = BCTrainer(
        mock_brain, trainer_parameters, training=True, load=False, seed=0, run_id=0
    )
    trainer.demonstration_buffer = mb.simulate_rollout(env, trainer.policy, 100)
    return trainer, env


def test_bc_trainer_step(dummy_config):
    trainer, env = create_bc_trainer(dummy_config)
    # Test get_step
    assert trainer.get_step == 0
    # Test update policy
    trainer.update_policy()
    assert len(trainer.stats["Losses/Cloning Loss"]) > 0
    # Test increment step
    trainer.increment_step(1)
    assert trainer.step == 1


def test_bc_trainer_add_proc_experiences(dummy_config):
    trainer, env = create_bc_trainer(dummy_config)
    # Test add_experiences
    returned_braininfo = env.step()
    trainer.add_experiences(
        returned_braininfo, returned_braininfo, {}
    )  # Take action outputs is not used
    for agent_id in returned_braininfo["Ball3DBrain"].agents:
        assert trainer.evaluation_buffer[agent_id].last_brain_info is not None
        assert trainer.episode_steps[agent_id] > 0
        assert trainer.cumulative_rewards[agent_id] > 0
    # Test process_experiences by setting done
    returned_braininfo["Ball3DBrain"].local_done = 12 * [True]
    trainer.process_experiences(returned_braininfo, returned_braininfo)
    for agent_id in returned_braininfo["Ball3DBrain"].agents:
        assert trainer.episode_steps[agent_id] == 0
        assert trainer.cumulative_rewards[agent_id] == 0


def test_bc_trainer_end_episode(dummy_config):
    trainer, env = create_bc_trainer(dummy_config)
    returned_braininfo = env.step()
    trainer.add_experiences(
        returned_braininfo, returned_braininfo, {}
    )  # Take action outputs is not used
    trainer.process_experiences(returned_braininfo, returned_braininfo)
    # Should set everything to 0
    trainer.end_episode()
    for agent_id in returned_braininfo["Ball3DBrain"].agents:
        assert trainer.episode_steps[agent_id] == 0
        assert trainer.cumulative_rewards[agent_id] == 0


@mock.patch("mlagents.envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents.envs.environment.UnityEnvironment.get_communicator")
def test_bc_policy_evaluate(mock_communicator, mock_launcher, dummy_config):
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
    policy = BCPolicy(
        0, env.brains[env.external_brain_names[0]], trainer_parameters, False
    )
    run_out = policy.evaluate(brain_info)
    assert run_out["action"].shape == (3, 2)

    env.close()


def test_cc_bc_model():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = BehavioralCloningModel(
                make_brain_parameters(discrete_action=False, visual_inputs=0)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.policy]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
            }
            sess.run(run_list, feed_dict=feed_dict)
            # env.close()


def test_dc_bc_model():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = BehavioralCloningModel(
                make_brain_parameters(discrete_action=True, visual_inputs=0)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.action_probs]
            feed_dict = {
                model.batch_size: 2,
                model.dropout_rate: 1.0,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.action_masks: np.ones([2, 2]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_visual_dc_bc_model():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = BehavioralCloningModel(
                make_brain_parameters(discrete_action=True, visual_inputs=2)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.action_probs]
            feed_dict = {
                model.batch_size: 2,
                model.dropout_rate: 1.0,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.visual_in[0]: np.ones([2, 40, 30, 3]),
                model.visual_in[1]: np.ones([2, 40, 30, 3]),
                model.action_masks: np.ones([2, 2]),
            }
            sess.run(run_list, feed_dict=feed_dict)


def test_visual_cc_bc_model():
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.variable_scope("FakeGraphScope"):
            model = BehavioralCloningModel(
                make_brain_parameters(discrete_action=False, visual_inputs=2)
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            run_list = [model.sample_action, model.policy]
            feed_dict = {
                model.batch_size: 2,
                model.sequence_length: 1,
                model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                model.visual_in[0]: np.ones([2, 40, 30, 3]),
                model.visual_in[1]: np.ones([2, 40, 30, 3]),
            }
            sess.run(run_list, feed_dict=feed_dict)


if __name__ == "__main__":
    pytest.main()
