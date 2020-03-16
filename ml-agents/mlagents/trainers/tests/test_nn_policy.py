import pytest

import numpy as np
from mlagents.tf_utils import tf

import yaml

from mlagents.trainers.policy.nn_policy import NNPolicy
from mlagents.trainers.models import EncoderType, ModelUtils
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.brain import BrainParameters, CameraResolution
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory


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
        normalize: true
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


def create_policy_mock(dummy_config, use_rnn, use_discrete, use_visual):
    mock_brain = mb.setup_mock_brain(
        use_discrete,
        use_visual,
        vector_action_space=VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
        discrete_action_space=DISCRETE_ACTION_SPACE,
    )

    trainer_parameters = dummy_config
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["use_recurrent"] = use_rnn
    policy = NNPolicy(0, mock_brain, trainer_parameters, False, False)
    return policy


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_policy_evaluate(dummy_config, rnn, visual, discrete):
    # Test evaluate
    tf.reset_default_graph()
    policy = create_policy_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    step = mb.create_batchedstep_from_brainparams(policy.brain, num_agents=NUM_AGENTS)

    run_out = policy.evaluate(step, list(step.agent_id))
    if discrete:
        run_out["action"].shape == (NUM_AGENTS, len(DISCRETE_ACTION_SPACE))
    else:
        assert run_out["action"].shape == (NUM_AGENTS, VECTOR_ACTION_SPACE[0])


def test_normalization(dummy_config):
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

    time_horizon = 6
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        vec_obs_size=1,
        num_vis_obs=0,
        action_space=[2],
    )
    # Change half of the obs to 0
    for i in range(3):
        trajectory.steps[i].obs[0] = np.zeros(1, dtype=np.float32)
    policy = policy = NNPolicy(0, brain_params, dummy_config, False, False)

    trajectory_buffer = trajectory.to_agentbuffer()
    policy.update_normalization(trajectory_buffer["vector_obs"])

    # Check that the running mean and variance is correct
    steps, mean, variance = policy.sess.run(
        [policy.normalization_steps, policy.running_mean, policy.running_variance]
    )

    assert steps == 6
    assert mean[0] == 0.5
    # Note: variance is divided by number of steps, and initialized to 1 to avoid
    # divide by 0. The right answer is 0.25
    assert (variance[0] - 1) / steps == 0.25

    # Make another update, this time with all 1's
    time_horizon = 10
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        vec_obs_size=1,
        num_vis_obs=0,
        action_space=[2],
    )
    trajectory_buffer = trajectory.to_agentbuffer()
    policy.update_normalization(trajectory_buffer["vector_obs"])

    # Check that the running mean and variance is correct
    steps, mean, variance = policy.sess.run(
        [policy.normalization_steps, policy.running_mean, policy.running_variance]
    )

    assert steps == 16
    assert mean[0] == 0.8125
    assert (variance[0] - 1) / steps == pytest.approx(0.152, abs=0.01)


def test_min_visual_size():
    # Make sure each EncoderType has an entry in MIS_RESOLUTION_FOR_ENCODER
    assert set(ModelUtils.MIN_RESOLUTION_FOR_ENCODER.keys()) == set(EncoderType)

    for encoder_type in EncoderType:
        with tf.Graph().as_default():
            good_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type]
            good_res = CameraResolution(
                width=good_size, height=good_size, num_channels=3
            )
            vis_input = ModelUtils.create_visual_input(good_res, "test_min_visual_size")
            ModelUtils._check_resolution_for_encoder(vis_input, encoder_type)
            enc_func = ModelUtils.get_encoder_for_type(encoder_type)
            enc_func(vis_input, 32, ModelUtils.swish, 1, "test", False)

        # Anything under the min size should raise an exception. If not, decrease the min size!
        with pytest.raises(Exception):
            with tf.Graph().as_default():
                bad_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type] - 1
                bad_res = CameraResolution(
                    width=bad_size, height=bad_size, num_channels=3
                )
                vis_input = ModelUtils.create_visual_input(
                    bad_res, "test_min_visual_size"
                )

                with pytest.raises(UnityTrainerException):
                    # Make sure we'd hit a friendly error during model setup time.
                    ModelUtils._check_resolution_for_encoder(vis_input, encoder_type)

                enc_func = ModelUtils.get_encoder_for_type(encoder_type)
                enc_func(vis_input, 32, ModelUtils.swish, 1, "test", False)


if __name__ == "__main__":
    pytest.main()
