from collections import defaultdict
from typing import List, Dict
import numpy as np
import abc

from mlagents.tf_utils import tf


class StatsWriter(abc.ABC):
    @abc.abstractmethod
    def write_stats(self, category: str, key: str, value: float, step: int) -> None:
        pass


class TensorboardWriter(StatsWriter):
    def __init__(self):
        self.summary_writers = {}

    def write_stats(self, category: str, key: str, value: float, step: int) -> None:
        if category not in self.summary_writers:
            self.summary_writers[category] = tf.summary.FileWriter(category)
        summary = tf.Summary()
        summary.value.add(tag="{}".format(key), simple_value=value)
        self.summary_writers[category].add_summary(summary, step)
        self.summary_writers[category].flush()


class StatsReporter:
    def __init__(self, writers: List[StatsWriter]):
        """
        Generic StatsReporter. A category is the broadest type of storage (would correspond the run name and trainer
        name, e.g. 3DBalltest_3DBall. A key is the type of stat it is (e.g. Environment/Reward). Finally the Value
        is the float value attached to this stat.
        """
        self.writers = writers
        self.stats_dict: Dict[str, Dict[str, List]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add_stat(self, key: str, value: float, category: str) -> None:
        self.stats_dict[category][key].append(value)

    @abc.abstractmethod
    def write_stats(self, category: str, step: int) -> None:
        for key in self.stats_dict[category]:
            if len(self.stats_dict[category][key]) > 0:
                stat_mean = float(np.mean(self.stats_dict[category][key]))
                self.stats_dict[category][key] = []
                for writer in self.writers:
                    writer.write_stats(category, key, stat_mean, step)
