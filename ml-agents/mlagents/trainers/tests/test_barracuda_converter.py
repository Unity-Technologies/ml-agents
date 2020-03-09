import os
import tempfile
import pytest
import yaml

import mlagents.trainers.tensorflow_to_barracuda as tf2bc
from mlagents.trainers.tests.test_nn_policy import create_policy_mock
from mlagents.tf_utils import tf
from mlagents.model_serialization import SerializationSettings, export_policy_model


def test_barracuda_converter():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    tmpfile = os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()) + ".nn"
    )

    # make sure there are no left-over files
    if os.path.isfile(tmpfile):
        os.remove(tmpfile)

    tf2bc.convert(path_prefix + "/BasicLearning.pb", tmpfile)

    # test if file exists after conversion
    assert os.path.isfile(tmpfile)
    # currently converter produces small output file even if input file is empty
    # 100 bytes is high enough to prove that conversion was successful
    assert os.path.getsize(tmpfile) > 100

    # cleanup
    os.remove(tmpfile)


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


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_policy_conversion(dummy_config, tmpdir, rnn, visual, discrete):
    tf.reset_default_graph()
    dummy_config["summary_path"] = str(tmpdir)
    dummy_config["model_path"] = os.path.join(tmpdir, "test")
    policy = create_policy_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    policy.save_model(1000)
    settings = SerializationSettings(
        policy.model_path, os.path.join(tmpdir, policy.brain.brain_name)
    )
    export_policy_model(settings, policy.graph, policy.sess)

    # These checks taken from test_barracuda_converter
    assert os.path.isfile(os.path.join(tmpdir, "test.nn"))
    assert os.path.getsize(os.path.join(tmpdir, "test.nn")) > 100
