import os
import unittest
import json

from mlagents.trainers.training_status import (
    StatusType,
    StatusMetaData,
    GlobalTrainingStatus,
)


def test_stats_reporter_store_restore(tmpdir):
    trainingstatus = GlobalTrainingStatus("Category1")
    path_dir = os.path.join(tmpdir, "test.json")

    trainingstatus.store_parameter_state(StatusType.LESSON_NUM, 3)
    GlobalTrainingStatus.save_state(path_dir)

    with open(path_dir, "r") as fp:
        test_json = json.load(fp)

    assert "Category1" in test_json
    assert StatusType.LESSON_NUM.value in test_json["Category1"]
    assert test_json["Category1"][StatusType.LESSON_NUM.value] == 3
    assert "metadata" in test_json

    statsreporter_new = GlobalTrainingStatus("Category1")
    GlobalTrainingStatus.load_state(path_dir)
    restored_val = statsreporter_new.restore_parameter_state(StatusType.LESSON_NUM)
    assert restored_val == 3


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
