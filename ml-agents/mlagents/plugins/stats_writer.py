from typing import List

try:
    # importlib.metadata is new in python3.8
    import importlib.metadata as importlib_metadata
except ImportError:
    # We use the backport for older python versions.
    import importlib_metadata  # type: ignore
from mlagents.trainers.stats import StatsWriter

from mlagents_envs import logging_util


logger = logging_util.get_logger(__name__)


def get_default_stats_writers() -> List[StatsWriter]:
    # TODO move construction of default StatsWriters here
    return []


def register_stats_writer_plugins() -> List[StatsWriter]:
    all_stats_writers: List[StatsWriter] = []
    eps = importlib_metadata.entry_points()["mlagents.stats_writer"]  # type: ignore

    for ep in eps:
        if ep.name != "default":
            logger.info(f"Initializing StatsWriter plugins: {ep.name}")

        try:
            plugin_func = ep.load()
            plugin_stats_writers = plugin_func()
            logger.debug(
                f"Found {len(plugin_stats_writers)} StatsWriters for plugin {ep.name}"
            )
            all_stats_writers += plugin_stats_writers
        except BaseException:
            # Catch all exceptions from setting up the plugin, so that bad user code doesn't break things.
            logger.exception(
                f"Error initializing StatsWriter plugins for {ep.name}. This plugin will not be used."
            )
    return all_stats_writers
