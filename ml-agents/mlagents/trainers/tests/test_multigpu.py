from unittest import mock
import pytest

from mlagents.tf_utils import tf
import yaml

from mlagents.trainers.ppo.multi_gpu_policy import MultiGpuPPOPolicy
from mlagents.trainers.tests.mock_brain import create_mock_brainparams


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
        memory_size: 8
        curiosity_strength: 0.0
        curiosity_enc_size: 1
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


@mock.patch("mlagents.trainers.ppo.multi_gpu_policy.get_devices")
def test_create_model(mock_get_devices, dummy_config):
    tf.reset_default_graph()
    mock_get_devices.return_value = [
        "/device:GPU:0",
        "/device:GPU:1",
        "/device:GPU:2",
        "/device:GPU:3",
    ]

    trainer_parameters = dummy_config
    trainer_parameters["model_path"] = ""
    trainer_parameters["keep_checkpoints"] = 3
    brain = create_mock_brainparams()

    policy = MultiGpuPPOPolicy(0, brain, trainer_parameters, False, False)
    assert len(policy.towers) == len(mock_get_devices.return_value)


@mock.patch("mlagents.trainers.ppo.multi_gpu_policy.get_devices")
def test_average_gradients(mock_get_devices, dummy_config):
    tf.reset_default_graph()
    mock_get_devices.return_value = [
        "/device:GPU:0",
        "/device:GPU:1",
        "/device:GPU:2",
        "/device:GPU:3",
    ]

    trainer_parameters = dummy_config
    trainer_parameters["model_path"] = ""
    trainer_parameters["keep_checkpoints"] = 3
    brain = create_mock_brainparams()
    with tf.Session() as sess:
        policy = MultiGpuPPOPolicy(0, brain, trainer_parameters, False, False)
        var = tf.Variable(0)
        tower_grads = [
            [(tf.constant(0.1), var)],
            [(tf.constant(0.2), var)],
            [(tf.constant(0.3), var)],
            [(tf.constant(0.4), var)],
        ]
        avg_grads = policy.average_gradients(tower_grads)

        init = tf.global_variables_initializer()
        sess.run(init)
        run_out = sess.run(avg_grads)
    assert run_out == [(0.25, 0)]


@mock.patch("mlagents.trainers.tf_policy.TFPolicy._execute_model")
@mock.patch("mlagents.trainers.ppo.policy.PPOPolicy.construct_feed_dict")
@mock.patch("mlagents.trainers.ppo.multi_gpu_policy.get_devices")
def test_update(
    mock_get_devices, mock_construct_feed_dict, mock_execute_model, dummy_config
):
    tf.reset_default_graph()
    mock_get_devices.return_value = ["/device:GPU:0", "/device:GPU:1"]
    mock_construct_feed_dict.return_value = {}
    mock_execute_model.return_value = {
        "value_loss": 0.1,
        "policy_loss": 0.3,
        "update_batch": None,
    }

    trainer_parameters = dummy_config
    trainer_parameters["model_path"] = ""
    trainer_parameters["keep_checkpoints"] = 3
    brain = create_mock_brainparams()
    policy = MultiGpuPPOPolicy(0, brain, trainer_parameters, False, False)
    mock_mini_batch = mock.Mock()
    mock_mini_batch.items.return_value = [("action", [1, 2]), ("value", [3, 4])]
    run_out = policy.update(mock_mini_batch, 1)

    assert mock_mini_batch.items.call_count == len(mock_get_devices.return_value)
    assert mock_construct_feed_dict.call_count == len(mock_get_devices.return_value)
    assert run_out["Losses/Value Loss"] == 0.1
    assert run_out["Losses/Policy Loss"] == 0.3


if __name__ == "__main__":
    pytest.main()
