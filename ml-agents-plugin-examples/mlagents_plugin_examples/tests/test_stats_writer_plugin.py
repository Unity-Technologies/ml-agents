import pytest

from mlagents.plugins.stats_writer import register_stats_writer_plugins
from mlagents.trainers.settings import RunOptions

from mlagents_plugin_examples.example_stats_writer import ExampleStatsWriter


@pytest.mark.slow
def test_register_stats_writers():
    # Make sure that the ExampleStatsWriter gets returned from the list of all StatsWriters
    stats_writers = register_stats_writer_plugins(RunOptions())
    assert any(isinstance(sw, ExampleStatsWriter) for sw in stats_writers)
