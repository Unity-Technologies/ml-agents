import unittest.mock as mock
import os
import pytest
import tempfile

from mlagents.trainers.stats import StatsReporter, TensorboardWriter


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
    mock_writer1.write_stats.assert_called_once_with("category1", "key1", 4.5, step)
    mock_writer2.write_stats.assert_called_once_with("category1", "key1", 4.5, step)


def test_stat_reporter_text():
    # Test add_writer
    mock_writer = mock.Mock()
    StatsReporter.writers.clear()
    StatsReporter.add_writer(mock_writer)
    assert len(StatsReporter.writers) == 1

    statsreporter1 = StatsReporter("category1")

    # Test write_text
    step = 10
    statsreporter1.write_text("this is a text", step)
    mock_writer.write_text.assert_called_once_with("category1", "this is a text", step)


@mock.patch("mlagents.tf_utils.tf.Summary")
@mock.patch("mlagents.tf_utils.tf.summary.FileWriter")
def test_tensorboard_writer(mock_filewriter, mock_summary):
    # Test write_stats
    category = "category1"
    with tempfile.TemporaryDirectory(prefix="unittest-") as base_dir:
        tb_writer = TensorboardWriter(base_dir)
        tb_writer.write_stats("category1", "key1", 1.0, 10)

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
