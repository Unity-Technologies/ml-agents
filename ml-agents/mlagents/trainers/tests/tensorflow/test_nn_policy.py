import pytest

import numpy as np
from mlagents.tf_utils import tf

from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.tf.models import ModelUtils, Tensor3DShape
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.settings import TrainerSettings, NetworkSettings, EncoderType
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 32
NUM_AGENTS = 12
EPSILON = 1e-7


def create_policy_mock(
    dummy_config: TrainerSettings,
    use_rnn: bool = False,
    use_discrete: bool = True,
    use_visual: bool = False,
    seed: int = 0,
) -> TFPolicy:
    mock_spec = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )

    trainer_settings = dummy_config
    trainer_settings.keep_checkpoints = 3
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings() if use_rnn else None
    )
    policy = TFPolicy(seed, mock_spec, trainer_settings)
    return policy


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
def test_policy_evaluate(rnn, visual, discrete):
    # Test evaluate
    tf.reset_default_graph()
    policy = create_policy_mock(
        TrainerSettings(), use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    decision_step, terminal_step = mb.create_steps_from_behavior_spec(
        policy.behavior_spec, num_agents=NUM_AGENTS
    )

    run_out = policy.evaluate(decision_step, list(decision_step.agent_id))
    if discrete:
        run_out["action"].shape == (NUM_AGENTS, len(DISCRETE_ACTION_SPACE))
    else:
        assert run_out["action"].shape == (NUM_AGENTS, VECTOR_ACTION_SPACE)


def test_large_normalization():
    behavior_spec = mb.setup_test_behavior_specs(
        use_discrete=True, use_visual=False, vector_action_space=[2], vector_obs_space=1
    )
    # Taken from Walker seed 3713 which causes NaN without proper initialization
    large_obs1 = [
        1800.00036621,
        1799.96972656,
        1800.01245117,
        1800.07214355,
        1800.02758789,
        1799.98303223,
        1799.88647461,
        1799.89575195,
        1800.03479004,
        1800.14025879,
        1800.17675781,
        1800.20581055,
        1800.33740234,
        1800.36450195,
        1800.43457031,
        1800.45544434,
        1800.44604492,
        1800.56713867,
        1800.73901367,
    ]
    large_obs2 = [
        1799.99975586,
        1799.96679688,
        1799.92980957,
        1799.89550781,
        1799.93774414,
        1799.95300293,
        1799.94067383,
        1799.92993164,
        1799.84057617,
        1799.69873047,
        1799.70605469,
        1799.82849121,
        1799.85095215,
        1799.76977539,
        1799.78283691,
        1799.76708984,
        1799.67163086,
        1799.59191895,
        1799.5135498,
        1799.45556641,
        1799.3717041,
    ]
    policy = TFPolicy(
        0,
        behavior_spec,
        TrainerSettings(network_settings=NetworkSettings(normalize=True)),
        "testdir",
        False,
    )
    time_horizon = len(large_obs1)
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_shapes=[(1,)],
        action_spec=behavior_spec.action_spec,
    )
    for i in range(time_horizon):
        trajectory.steps[i].obs[0] = np.array([large_obs1[i]], dtype=np.float32)
    trajectory_buffer = trajectory.to_agentbuffer()
    policy.update_normalization(trajectory_buffer["vector_obs"])

    # Check that the running mean and variance is correct
    steps, mean, variance = policy.sess.run(
        [policy.normalization_steps, policy.running_mean, policy.running_variance]
    )
    assert mean[0] == pytest.approx(np.mean(large_obs1, dtype=np.float32), abs=0.01)
    assert variance[0] / steps == pytest.approx(
        np.var(large_obs1, dtype=np.float32), abs=0.01
    )

    time_horizon = len(large_obs2)
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_shapes=[(1,)],
        action_spec=behavior_spec.action_spec,
    )
    for i in range(time_horizon):
        trajectory.steps[i].obs[0] = np.array([large_obs2[i]], dtype=np.float32)

    trajectory_buffer = trajectory.to_agentbuffer()
    policy.update_normalization(trajectory_buffer["vector_obs"])

    steps, mean, variance = policy.sess.run(
        [policy.normalization_steps, policy.running_mean, policy.running_variance]
    )

    assert mean[0] == pytest.approx(
        np.mean(large_obs1 + large_obs2, dtype=np.float32), abs=0.01
    )
    assert variance[0] / steps == pytest.approx(
        np.var(large_obs1 + large_obs2, dtype=np.float32), abs=0.01
    )


def test_normalization():
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
    policy = TFPolicy(
        0,
        behavior_spec,
        TrainerSettings(network_settings=NetworkSettings(normalize=True)),
        "testdir",
        False,
    )

    trajectory_buffer = trajectory.to_agentbuffer()
    policy.update_normalization(trajectory_buffer["vector_obs"])

    # Check that the running mean and variance is correct
    steps, mean, variance = policy.sess.run(
        [policy.normalization_steps, policy.running_mean, policy.running_variance]
    )

    assert steps == 6
    assert mean[0] == 0.5
    # Note: variance is initalized to the variance of the initial trajectory + EPSILON
    # (to avoid divide by 0) and multiplied by the number of steps. The correct answer is 0.25
    assert variance[0] / steps == pytest.approx(0.25, abs=0.01)
    # Make another update, this time with all 1's
    time_horizon = 10
    trajectory = make_fake_trajectory(
        length=time_horizon,
        max_step_complete=True,
        observation_shapes=[(1,)],
        action_spec=behavior_spec.action_spec,
    )
    trajectory_buffer = trajectory.to_agentbuffer()
    policy.update_normalization(trajectory_buffer["vector_obs"])

    # Check that the running mean and variance is correct
    steps, mean, variance = policy.sess.run(
        [policy.normalization_steps, policy.running_mean, policy.running_variance]
    )

    assert steps == 16
    assert mean[0] == 0.8125
    assert variance[0] / steps == pytest.approx(0.152, abs=0.01)


def test_min_visual_size():
    # Make sure each EncoderType has an entry in MIS_RESOLUTION_FOR_ENCODER
    assert set(ModelUtils.MIN_RESOLUTION_FOR_ENCODER.keys()) == set(EncoderType)

    for encoder_type in EncoderType:
        with tf.Graph().as_default():
            good_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type]
            good_res = Tensor3DShape(width=good_size, height=good_size, num_channels=3)
            vis_input = ModelUtils.create_visual_input(good_res, "test_min_visual_size")
            ModelUtils._check_resolution_for_encoder(vis_input, encoder_type)
            enc_func = ModelUtils.get_encoder_for_type(encoder_type)
            enc_func(vis_input, 32, ModelUtils.swish, 1, "test", False)

        # Anything under the min size should raise an exception. If not, decrease the min size!
        with pytest.raises(Exception):
            with tf.Graph().as_default():
                bad_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type] - 1
                bad_res = Tensor3DShape(width=bad_size, height=bad_size, num_channels=3)
                vis_input = ModelUtils.create_visual_input(
                    bad_res, "test_min_visual_size"
                )

                with pytest.raises(UnityTrainerException):
                    # Make sure we'd hit a friendly error during model setup time.
                    ModelUtils._check_resolution_for_encoder(vis_input, encoder_type)

                enc_func = ModelUtils.get_encoder_for_type(encoder_type)
                enc_func(vis_input, 32, ModelUtils.swish, 1, "test", False)


def test_step_overflow():
    behavior_spec = mb.setup_test_behavior_specs(
        use_discrete=True, use_visual=False, vector_action_space=[2], vector_obs_space=1
    )

    policy = TFPolicy(
        0,
        behavior_spec,
        TrainerSettings(network_settings=NetworkSettings(normalize=True)),
        create_tf_graph=False,
    )
    policy.create_input_placeholders()
    policy.initialize()

    policy.set_step(2 ** 31 - 1)
    assert policy.get_current_step() == 2 ** 31 - 1
    policy.increment_step(3)
    assert policy.get_current_step() == 2 ** 31 + 2


if __name__ == "__main__":
    pytest.main()
