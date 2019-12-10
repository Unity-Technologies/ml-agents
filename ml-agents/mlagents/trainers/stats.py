from collections import defaultdict
import numpy as np
import abc

from mlagents.tf_utils import tf


class StatsReporter(abc.ABC):
    @abc.abstractmethod
    def add_stat(self, key: str, value: float, category: str) -> None:
        pass

    @abc.abstractmethod
    def write_stats(self, category: str, step: int) -> None:
        pass


class TensorboardReporter(StatsReporter):
    def __init__(self):
        self.summary_writers = {}
        self.stats_dict = defaultdict(defaultdict(list))

    def add_stat(self, key: str, value: float, category: str) -> None:
        if category not in self.summary_writers:
            self.summary_writers[category] = tf.summary.FileWriter(category)
        self.stats_dict[category][key].append(value)

    def write_stats(self, category: str, step: int) -> None:
        if category not in self.summary_writers:
            self.summary_writers[category] = tf.summary.FileWriter(category)
        summary = tf.Summary()
        for key in self.stats_dict[category]:
            if len(self.stats_dict[category][key]) > 0:
                stat_mean = float(np.mean(self.stats_dict[category][key]))
                summary.value.add(tag="{}".format(key), simple_value=stat_mean)
                self.stats_dict[category][key] = []
        self.summary_writers[category].add_summary(summary, step)
        self.summary_writers[category].flush()
