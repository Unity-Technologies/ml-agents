from typing import Dict, List
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.stats import StatsWriter, StatsSummary


class ExampleStatsWriter(StatsWriter):
    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        print(f"ExampleStatsWriter category: {category} values: {values}")


def get_example_stats_writer(run_options: RunOptions) -> List[StatsWriter]:
    print("Creating a new stats writer! This is so exciting!")
    return [ExampleStatsWriter()]
