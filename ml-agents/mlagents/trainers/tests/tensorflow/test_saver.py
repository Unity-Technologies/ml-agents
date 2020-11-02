import pytest
from unittest import mock
import os
import unittest
import tempfile

import numpy as np
from mlagents.tf_utils import tf
from mlagents.trainers.model_saver.tf_model_saver import TFModelSaver
from mlagents.trainers import __version__
from mlagents.trainers.settings import TrainerSettings, NetworkSettings
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.tensorflow.test_nn_policy import create_policy_mock
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.ppo.optimizer_tf import PPOOptimizer


def test_register(tmp_path):
    trainer_params = TrainerSettings()
    model_saver = TFModelSaver(trainer_params, tmp_path)

    opt = mock.Mock(spec=PPOOptimizer)
    model_saver.register(opt)
    assert model_saver.policy is None

    trainer_params = TrainerSettings()
    policy = create_policy_mock(trainer_params)
    model_saver.register(policy)
    assert model_saver.policy is not None


class ModelVersionTest(unittest.TestCase):
    def test_version_compare(self):
        # Test write_stats
        with self.assertLogs("mlagents.trainers", level="WARNING") as cm:
            trainer_params = TrainerSettings()
            mock_path = tempfile.mkdtemp()
            policy = create_policy_mock(trainer_params)
            model_saver = TFModelSaver(trainer_params, mock_path)
            model_saver.register(policy)

            model_saver._check_model_version(
                "0.0.0"
            )  # This is not the right version for sure
            # Assert that 1 warning has been thrown with incorrect version
            assert len(cm.output) == 1
            model_saver._check_model_version(
                __version__
            )  # This should be the right version
            # Assert that no additional warnings have been thrown wth correct ver
            assert len(cm.output) == 1


def test_load_save(tmp_path):
    path1 = os.path.join(tmp_path, "runid1")
    path2 = os.path.join(tmp_path, "runid2")
    trainer_params = TrainerSettings()
    policy = create_policy_mock(trainer_params)
    model_saver = TFModelSaver(trainer_params, path1)
    model_saver.register(policy)
    model_saver.initialize_or_load(policy)
    policy.set_step(2000)

    mock_brain_name = "MockBrain"
    model_saver.save_checkpoint(mock_brain_name, 2000)
    assert len(os.listdir(tmp_path)) > 0

    # Try load from this path
    model_saver = TFModelSaver(trainer_params, path1, load=True)
    policy2 = create_policy_mock(trainer_params)
    model_saver.register(policy2)
    model_saver.initialize_or_load(policy2)
    _compare_two_policies(policy, policy2)
    assert policy2.get_current_step() == 2000

    # Try initialize from path 1
    trainer_params.init_path = path1
    model_saver = TFModelSaver(trainer_params, path2)
    policy3 = create_policy_mock(trainer_params)
    model_saver.register(policy3)
    model_saver.initialize_or_load(policy3)

    _compare_two_policies(policy2, policy3)
    # Assert that the steps are 0.
    assert policy3.get_current_step() == 0


def _compare_two_policies(policy1: TFPolicy, policy2: TFPolicy) -> None:
    """
    Make sure two policies have the same output for the same input.
    """
    decision_step, _ = mb.create_steps_from_behavior_spec(
        policy1.behavior_spec, num_agents=1
    )
    run_out1 = policy1.evaluate(decision_step, list(decision_step.agent_id))
    run_out2 = policy2.evaluate(decision_step, list(decision_step.agent_id))

    np.testing.assert_array_equal(run_out2["log_probs"], run_out1["log_probs"])


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_checkpoint_conversion(tmpdir, rnn, visual, discrete):
    tf.reset_default_graph()
    dummy_config = TrainerSettings()
    model_path = os.path.join(tmpdir, "Mock_Brain")
    policy = create_policy_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    trainer_params = TrainerSettings()
    model_saver = TFModelSaver(trainer_params, model_path)
    model_saver.register(policy)
    model_saver.save_checkpoint("Mock_Brain", 100)
    assert os.path.isfile(model_path + "/Mock_Brain-100.nn")


# This is the normalizer test from test_nn_policy.py but with a load
def test_normalizer_after_load(tmp_path):
    behavior_spec = mb.setup_test_behavior_specs(
        use_discrete=True, use_visual=False, vector_action_space=[2], vector_obs_space=1
    )
    time_horizon = 6
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_shapes=[(1,)],
        action_spec=behavior_spec.action_spec,
    )
    # Change half of the obs to 0
    for i in range(3):
        trajectory.steps[i].obs[0] = np.zeros(1, dtype=np.float32)

    trainer_params = TrainerSettings(network_settings=NetworkSettings(normalize=True))
    policy = TFPolicy(0, behavior_spec, trainer_params)

    trajectory_buffer = trajectory.to_agentbuffer()
    policy.update_normalization(trajectory_buffer["vector_obs"])

    # Check that the running mean and variance is correct
    steps, mean, variance = policy.sess.run(
        [policy.normalization_steps, policy.running_mean, policy.running_variance]
    )

    assert steps == 6
    assert mean[0] == 0.5
    assert variance[0] / steps == pytest.approx(0.25, abs=0.01)
    # Save ckpt and load into another policy
    path1 = os.path.join(tmp_path, "runid1")
    model_saver = TFModelSaver(trainer_params, path1)
    model_saver.register(policy)
    mock_brain_name = "MockBrain"
    model_saver.save_checkpoint(mock_brain_name, 6)
    assert len(os.listdir(tmp_path)) > 0
    policy1 = TFPolicy(0, behavior_spec, trainer_params)
    model_saver = TFModelSaver(trainer_params, path1, load=True)
    model_saver.register(policy1)
    model_saver.initialize_or_load(policy1)

    # Make another update to new policy, this time with all 1's
    time_horizon = 10
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_shapes=[(1,)],
        action_spec=behavior_spec.action_spec,
    )
    trajectory_buffer = trajectory.to_agentbuffer()
    policy1.update_normalization(trajectory_buffer["vector_obs"])

    # Check that the running mean and variance is correct
    steps, mean, variance = policy1.sess.run(
        [policy1.normalization_steps, policy1.running_mean, policy1.running_variance]
    )

    assert steps == 16
    assert mean[0] == 0.8125
    assert variance[0] / steps == pytest.approx(0.152, abs=0.01)
