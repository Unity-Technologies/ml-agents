from typing import Dict, List
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.stats import StatsWriter, StatsSummary


class ExampleStatsWriter(StatsWriter):
    """
    Example implementation of the StatsWriter abstract class.
    This doesn't do anything interesting, just prints the stats that it gets.
    """

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        print(f"ExampleStatsWriter category: {category} values: {values}")


def get_example_stats_writer(run_options: RunOptions) -> List[StatsWriter]:
    """
    Registration function. This is referenced in setup.py and will
    be called by mlagents-learn when it starts to determine the
    list of StatsWriters to use.

    It must return a list of StatsWriters.
    """
    print("Creating a new stats writer! This is so exciting!")
    return [ExampleStatsWriter()]
