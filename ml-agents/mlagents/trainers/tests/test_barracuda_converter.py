import os
import yaml
import pytest
import tempfile

import mlagents.trainers.tensorflow_to_barracuda as tf2bc
from mlagents.trainers.tests.test_bc import create_bc_trainer


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
def bc_dummy_config():
    return yaml.safe_load(
        """
            hidden_units: 32
            learning_rate: 3.0e-4
            num_layers: 1
            use_recurrent: false
            sequence_length: 32
            memory_size: 64
            batches_per_epoch: 1
            batch_size: 64
            summary_freq: 2000
            max_steps: 4000
            """
    )


@pytest.mark.parametrize("use_lstm", [False, True], ids=["nolstm", "lstm"])
@pytest.mark.parametrize("use_discrete", [True, False], ids=["disc", "cont"])
def test_bc_export(bc_dummy_config, use_lstm, use_discrete):
    bc_dummy_config["use_recurrent"] = use_lstm
    trainer, env = create_bc_trainer(bc_dummy_config, use_discrete)
    trainer.export_model()
