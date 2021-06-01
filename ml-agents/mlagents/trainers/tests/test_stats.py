from unittest import mock
import os
import pytest
import tempfile
import unittest
import time


from mlagents.trainers.stats import (
    StatsReporter,
    TensorboardWriter,
    StatsSummary,
    GaugeWriter,
    ConsoleWriter,
    StatsPropertyType,
    StatsAggregationMethod,
)

from mlagents.trainers.env_manager import AgentManager


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

    statsreportercalls = [
        mock.call(f"category{j}", f"key{j}", float(i), StatsAggregationMethod.AVERAGE)
        for i in range(10)
        for j in [1, 2]
    ]

    mock_writer1.on_add_stat.assert_has_calls(statsreportercalls)
    mock_writer2.on_add_stat.assert_has_calls(statsreportercalls)

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


@mock.patch("mlagents.trainers.stats.SummaryWriter")
def test_tensorboard_writer(mock_summary):
    # Test write_stats
    category = "category1"
    with tempfile.TemporaryDirectory(prefix="unittest-") as base_dir:
        tb_writer = TensorboardWriter(base_dir, clear_past_data=False)
        statssummary1 = StatsSummary(
            full_dist=[1.0], aggregation_method=StatsAggregationMethod.AVERAGE
        )
        tb_writer.write_stats("category1", {"key1": statssummary1}, 10)

        # Test that the filewriter has been created and the directory has been created.
        filewriter_dir = "{basedir}/{category}".format(
            basedir=base_dir, category=category
        )
        assert os.path.exists(filewriter_dir)
        mock_summary.assert_called_once_with(filewriter_dir)

        # Test that the filewriter was written to and the summary was added.
        mock_summary.return_value.add_scalar.assert_called_once_with("key1", 1.0, 10)
        mock_summary.return_value.flush.assert_called_once()

        # Test hyperparameter writing - no good way to parse the TB string though.
        tb_writer.add_property(
            "category1", StatsPropertyType.HYPERPARAMETERS, {"example": 1.0}
        )
        assert mock_summary.return_value.add_text.call_count >= 1


@pytest.mark.parametrize("aggregation_type", list(StatsAggregationMethod))
def test_agent_manager_stats_report(aggregation_type):
    stats_reporter = StatsReporter("recorder_name")
    manager = AgentManager(None, "behaviorName", stats_reporter)

    values = range(5)

    env_stats = {"stat": [(i, aggregation_type) for i in values]}
    manager.record_environment_stats(env_stats, 0)
    summary = stats_reporter.get_stats_summaries("stat")
    aggregation_result = {
        StatsAggregationMethod.AVERAGE: sum(values) / len(values),
        StatsAggregationMethod.MOST_RECENT: values[-1],
        StatsAggregationMethod.SUM: sum(values),
        StatsAggregationMethod.HISTOGRAM: sum(values) / len(values),
    }

    assert summary.aggregated_value == aggregation_result[aggregation_type]
    stats_reporter.write_stats(0)


def test_tensorboard_writer_clear(tmp_path):
    tb_writer = TensorboardWriter(tmp_path, clear_past_data=False)
    statssummary1 = StatsSummary(
        full_dist=[1.0], aggregation_method=StatsAggregationMethod.AVERAGE
    )
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


@mock.patch("mlagents.trainers.stats.SummaryWriter")
def test_tensorboard_writer_hidden_keys(mock_summary):
    # Test write_stats
    category = "category1"
    with tempfile.TemporaryDirectory(prefix="unittest-") as base_dir:
        tb_writer = TensorboardWriter(
            base_dir, clear_past_data=False, hidden_keys="hiddenKey"
        )
        statssummary1 = StatsSummary(
            full_dist=[1.0], aggregation_method=StatsAggregationMethod.AVERAGE
        )
        tb_writer.write_stats("category1", {"hiddenKey": statssummary1}, 10)

        # Test that the filewriter has been created and the directory has been created.
        filewriter_dir = "{basedir}/{category}".format(
            basedir=base_dir, category=category
        )
        assert os.path.exists(filewriter_dir)
        mock_summary.assert_called_once_with(filewriter_dir)

        # Test that the filewriter was not written to since we used the hidden key.
        mock_summary.return_value.add_scalar.assert_not_called()
        mock_summary.return_value.flush.assert_not_called()


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
            statssummary1 = StatsSummary(
                full_dist=[1.0], aggregation_method=StatsAggregationMethod.AVERAGE
            )
            console_writer.write_stats(
                category,
                {
                    "Environment/Cumulative Reward": statssummary1,
                    "Is Training": statssummary1,
                },
                10,
            )
            statssummary2 = StatsSummary(
                full_dist=[0.0], aggregation_method=StatsAggregationMethod.AVERAGE
            )
            console_writer.write_stats(
                category,
                {
                    "Environment/Cumulative Reward": statssummary2,
                    "Is Training": statssummary2,
                },
                10,
            )
            # Test hyperparameter writing
            console_writer.add_property(
                "category1", StatsPropertyType.HYPERPARAMETERS, {"example": 1.0}
            )

        self.assertIn(
            "Mean Reward: 1.000. Std of Reward: 0.000. Training.", cm.output[0]
        )
        self.assertIn("Not Training.", cm.output[1])

        self.assertIn("Hyperparameters for behavior name", cm.output[2])
        self.assertIn("example:\t1.0", cm.output[2])

    def test_selfplay_console_writer(self):
        with self.assertLogs("mlagents.trainers", level="INFO") as cm:
            category = "category1"
            console_writer = ConsoleWriter()
            console_writer.add_property(category, StatsPropertyType.SELF_PLAY, True)
            statssummary1 = StatsSummary(
                full_dist=[1.0], aggregation_method=StatsAggregationMethod.AVERAGE
            )
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
            "Mean Reward: 1.000. Std of Reward: 0.000. Training.", cm.output[0]
        )
