from unittest import mock
import os
import pytest
import tempfile
import unittest
import csv
import time

from mlagents.trainers.stats import (
    StatsReporter,
    TensorboardWriter,
    CSVWriter,
    StatsSummary,
    GaugeWriter,
    ConsoleWriter,
    StatsPropertyType,
)


def test_stat_reporter_add_summary_write():
    # Test add_writer
    StatsReporter.writers.clear()
    mock_writer1 = mock.Mock()
    mock_writer2 = mock.Mock()
    StatsReporter.add_writer(mock_writer1)
    StatsReporter.add_writer(mock_writer2)
    assert len(StatsReporter.writers) == 2

    # Test add_stats and summaries
    statsreporter1 = StatsReporter("category1")
    statsreporter2 = StatsReporter("category2")
    for i in range(10):
        statsreporter1.add_stat("key1", float(i))
        statsreporter2.add_stat("key2", float(i))

    statssummary1 = statsreporter1.get_stats_summaries("key1")
    statssummary2 = statsreporter2.get_stats_summaries("key2")

    assert statssummary1.num == 10
    assert statssummary2.num == 10
    assert statssummary1.mean == 4.5
    assert statssummary2.mean == 4.5
    assert statssummary1.std == pytest.approx(2.9, abs=0.1)
    assert statssummary2.std == pytest.approx(2.9, abs=0.1)

    # Test write_stats
    step = 10
    statsreporter1.write_stats(step)
    mock_writer1.write_stats.assert_called_once_with(
        "category1", {"key1": statssummary1}, step
    )
    mock_writer2.write_stats.assert_called_once_with(
        "category1", {"key1": statssummary1}, step
    )


def test_stat_reporter_property():
    # Test add_writer
    mock_writer = mock.Mock()
    StatsReporter.writers.clear()
    StatsReporter.add_writer(mock_writer)
    assert len(StatsReporter.writers) == 1

    statsreporter1 = StatsReporter("category1")

    # Test add_property
    statsreporter1.add_property("key", "this is a text")
    mock_writer.add_property.assert_called_once_with(
        "category1", "key", "this is a text"
    )


@mock.patch("mlagents.tf_utils.tf.Summary")
@mock.patch("mlagents.tf_utils.tf.summary.FileWriter")
def test_tensorboard_writer(mock_filewriter, mock_summary):
    # Test write_stats
    category = "category1"
    with tempfile.TemporaryDirectory(prefix="unittest-") as base_dir:
        tb_writer = TensorboardWriter(base_dir, clear_past_data=False)
        statssummary1 = StatsSummary(mean=1.0, std=1.0, num=1)
        tb_writer.write_stats("category1", {"key1": statssummary1}, 10)

        # Test that the filewriter has been created and the directory has been created.
        filewriter_dir = "{basedir}/{category}".format(
            basedir=base_dir, category=category
        )
        assert os.path.exists(filewriter_dir)
        mock_filewriter.assert_called_once_with(filewriter_dir)

        # Test that the filewriter was written to and the summary was added.
        mock_summary.return_value.value.add.assert_called_once_with(
            tag="key1", simple_value=1.0
        )
        mock_filewriter.return_value.add_summary.assert_called_once_with(
            mock_summary.return_value, 10
        )
        mock_filewriter.return_value.flush.assert_called_once()

        # Test hyperparameter writing - no good way to parse the TB string though.
        tb_writer.add_property(
            "category1", StatsPropertyType.HYPERPARAMETERS, {"example": 1.0}
        )
        assert mock_filewriter.return_value.add_summary.call_count > 1


def test_tensorboard_writer_clear(tmp_path):
    tb_writer = TensorboardWriter(tmp_path, clear_past_data=False)
    statssummary1 = StatsSummary(mean=1.0, std=1.0, num=1)
    tb_writer.write_stats("category1", {"key1": statssummary1}, 10)
    # TB has some sort of timeout before making a new file
    time.sleep(1.0)
    assert len(os.listdir(os.path.join(tmp_path, "category1"))) > 0

    # See if creating a new one doesn't delete it
    tb_writer = TensorboardWriter(tmp_path, clear_past_data=False)
    tb_writer.write_stats("category1", {"key1": statssummary1}, 10)
    assert len(os.listdir(os.path.join(tmp_path, "category1"))) > 1
    time.sleep(1.0)

    # See if creating a new one deletes old ones
    tb_writer = TensorboardWriter(tmp_path, clear_past_data=True)
    tb_writer.write_stats("category1", {"key1": statssummary1}, 10)
    assert len(os.listdir(os.path.join(tmp_path, "category1"))) == 1


def test_csv_writer():
    # Test write_stats
    category = "category1"
    with tempfile.TemporaryDirectory(prefix="unittest-") as base_dir:
        csv_writer = CSVWriter(base_dir, required_fields=["key1", "key2"])
        statssummary1 = StatsSummary(mean=1.0, std=1.0, num=1)
        csv_writer.write_stats("category1", {"key1": statssummary1}, 10)

        # Test that the filewriter has been created and the directory has been created.
        filewriter_dir = "{basedir}/{category}.csv".format(
            basedir=base_dir, category=category
        )
        # The required keys weren't in the stats
        assert not os.path.exists(filewriter_dir)

        csv_writer.write_stats(
            "category1", {"key1": statssummary1, "key2": statssummary1}, 10
        )
        csv_writer.write_stats(
            "category1", {"key1": statssummary1, "key2": statssummary1}, 20
        )

        # The required keys were in the stats
        assert os.path.exists(filewriter_dir)

        with open(filewriter_dir) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    assert "key1" in row
                    assert "key2" in row
                    assert "Steps" in row
                    line_count += 1
                else:
                    assert len(row) == 3
                    line_count += 1
            assert line_count == 3


def test_gauge_stat_writer_sanitize():
    assert GaugeWriter.sanitize_string("Policy/Learning Rate") == "Policy.LearningRate"
    assert (
        GaugeWriter.sanitize_string("Very/Very/Very Nested Stat")
        == "Very.Very.VeryNestedStat"
    )


class ConsoleWriterTest(unittest.TestCase):
    def test_console_writer(self):
        # Test write_stats
        with self.assertLogs("mlagents.trainers", level="INFO") as cm:
            category = "category1"
            console_writer = ConsoleWriter()
            statssummary1 = StatsSummary(mean=1.0, std=1.0, num=1)
            console_writer.write_stats(
                category,
                {
                    "Environment/Cumulative Reward": statssummary1,
                    "Is Training": statssummary1,
                },
                10,
            )
            statssummary2 = StatsSummary(mean=0.0, std=0.0, num=1)
            console_writer.write_stats(
                category,
                {
                    "Environment/Cumulative Reward": statssummary1,
                    "Is Training": statssummary2,
                },
                10,
            )
            # Test hyperparameter writing - no good way to parse the TB string though.
            console_writer.add_property(
                "category1", StatsPropertyType.HYPERPARAMETERS, {"example": 1.0}
            )

        self.assertIn(
            "Mean Reward: 1.000. Std of Reward: 1.000. Training.", cm.output[0]
        )
        self.assertIn("Not Training.", cm.output[1])

        self.assertIn("Hyperparameters for behavior name", cm.output[2])
        self.assertIn("example:\t1.0", cm.output[2])

    def test_selfplay_console_writer(self):
        with self.assertLogs("mlagents.trainers", level="INFO") as cm:
            category = "category1"
            console_writer = ConsoleWriter()
            console_writer.add_property(category, StatsPropertyType.SELF_PLAY, True)
            statssummary1 = StatsSummary(mean=1.0, std=1.0, num=1)
            console_writer.write_stats(
                category,
                {
                    "Environment/Cumulative Reward": statssummary1,
                    "Is Training": statssummary1,
                    "Self-play/ELO": statssummary1,
                },
                10,
            )

        self.assertIn(
            "Mean Reward: 1.000. Std of Reward: 1.000. Training.", cm.output[0]
        )
