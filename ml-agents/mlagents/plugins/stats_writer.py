from typing import List
import importlib_metadata
from mlagents.trainers.stats import StatsWriter


def get_default_stats_writers() -> List[StatsWriter]:
    # TODO move construction of default StatsWriters here
    return []


def register_stats_writer_plugins() -> List[StatsWriter]:
    all_stats_writers: List[StatsWriter] = []
    eps = importlib_metadata.entry_points()["mlagents.stats_writer"]

    for ep in eps:
        print(f"registering {ep.name}")
        # TODO try/except around all of this
        plugin_func = ep.load()
        all_stats_writers += plugin_func()
    return all_stats_writers
