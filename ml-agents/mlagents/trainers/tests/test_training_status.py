import os
import unittest
import json
from enum import Enum
import time
from mlagents.trainers.training_status import (
    StatusType,
    StatusMetaData,
    GlobalTrainingStatus,
)
from mlagents.trainers.policy.checkpoint_manager import (
    NNCheckpointManager,
    NNCheckpoint,
)


def test_globaltrainingstatus(tmpdir):
    path_dir = os.path.join(tmpdir, "test.json")

    GlobalTrainingStatus.set_parameter_state("Category1", StatusType.LESSON_NUM, 3)
    GlobalTrainingStatus.save_state(path_dir)

    with open(path_dir) as fp:
        test_json = json.load(fp)

    assert "Category1" in test_json
    assert StatusType.LESSON_NUM.value in test_json["Category1"]
    assert test_json["Category1"][StatusType.LESSON_NUM.value] == 3
    assert "metadata" in test_json

    GlobalTrainingStatus.load_state(path_dir)
    restored_val = GlobalTrainingStatus.get_parameter_state(
        "Category1", StatusType.LESSON_NUM
    )
    assert restored_val == 3

    # Test unknown categories and status types (keys)
    unknown_category = GlobalTrainingStatus.get_parameter_state(
        "Category3", StatusType.LESSON_NUM
    )

    class FakeStatusType(Enum):
        NOTAREALKEY = "notarealkey"

    unknown_key = GlobalTrainingStatus.get_parameter_state(
        "Category1", FakeStatusType.NOTAREALKEY
    )
    assert unknown_category is None
    assert unknown_key is None


def test_model_management(tmpdir):

    results_path = os.path.join(tmpdir, "results")
    brain_name = "Mock_brain"
    final_model_path = os.path.join(results_path, brain_name)
    test_checkpoint_list = [
        {
            "steps": 1,
            "file_path": os.path.join(final_model_path, f"{brain_name}-1.nn"),
            "reward": 1.312,
            "creation_time": time.time(),
        },
        {
            "steps": 2,
            "file_path": os.path.join(final_model_path, f"{brain_name}-2.nn"),
            "reward": 1.912,
            "creation_time": time.time(),
        },
        {
            "steps": 3,
            "file_path": os.path.join(final_model_path, f"{brain_name}-3.nn"),
            "reward": 2.312,
            "creation_time": time.time(),
        },
    ]
    GlobalTrainingStatus.set_parameter_state(
        brain_name, StatusType.CHECKPOINTS, test_checkpoint_list
    )

    new_checkpoint_4 = NNCheckpoint(
        4, os.path.join(final_model_path, f"{brain_name}-4.nn"), 2.678, time.time()
    )
    NNCheckpointManager.add_checkpoint(brain_name, new_checkpoint_4, 4)
    assert len(NNCheckpointManager.get_checkpoints(brain_name)) == 4

    new_checkpoint_5 = NNCheckpoint(
        5, os.path.join(final_model_path, f"{brain_name}-5.nn"), 3.122, time.time()
    )
    NNCheckpointManager.add_checkpoint(brain_name, new_checkpoint_5, 4)
    assert len(NNCheckpointManager.get_checkpoints(brain_name)) == 4

    final_model_path = f"{final_model_path}.nn"
    final_model_time = time.time()
    current_step = 6
    final_model = NNCheckpoint(current_step, final_model_path, 3.294, final_model_time)

    NNCheckpointManager.track_final_checkpoint(brain_name, final_model)
    assert len(NNCheckpointManager.get_checkpoints(brain_name)) == 4

    check_checkpoints = GlobalTrainingStatus.saved_state[brain_name][
        StatusType.CHECKPOINTS.value
    ]
    assert check_checkpoints is not None

    final_model = GlobalTrainingStatus.saved_state[StatusType.FINAL_CHECKPOINT.value]
    assert final_model is not None


class StatsMetaDataTest(unittest.TestCase):
    def test_metadata_compare(self):
        # Test write_stats
        with self.assertLogs("mlagents.trainers", level="WARNING") as cm:
            default_metadata = StatusMetaData()
            version_statsmetadata = StatusMetaData(mlagents_version="test")
            default_metadata.check_compatibility(version_statsmetadata)

            tf_version_statsmetadata = StatusMetaData(tensorflow_version="test")
            default_metadata.check_compatibility(tf_version_statsmetadata)

        # Assert that 2 warnings have been thrown
        assert len(cm.output) == 2
