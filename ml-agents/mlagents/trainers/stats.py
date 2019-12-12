from collections import defaultdict
from typing import List, Dict, NamedTuple
import numpy as np
import abc

from mlagents.tf_utils import tf


class StatsWriter(abc.ABC):
    """
    A StatsWriter abstract class. A StatsWriter takes in a category, key, scalar value, and step
    and writes it out by some method.
    """

    @abc.abstractmethod
    def write_stats(self, category: str, key: str, value: float, step: int) -> None:
        pass

    @abc.abstractmethod
    def write_text(self, category: str, text: str, step: int) -> None:
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

    def write_text(self, category: str, text: str, step: int) -> None:
        if category not in self.summary_writers:
            self.summary_writers[category] = tf.summary.FileWriter(category)
        self.summary_writers[category].add_summary(text, step)


class StatsSummary(NamedTuple):
    mean: float
    std: float
    num: int


class StatsReporter:
    def __init__(self, writers: List[StatsWriter]):
        """
        Generic StatsReporter. A category is the broadest type of storage (would
        correspond the run name and trainer name, e.g. 3DBalltest_3DBall. A key is the
        type of stat it is (e.g. Environment/Reward). Finally the Value is the float value
        attached to this stat.
        """
        self.writers = writers
        self.stats_dict: Dict[str, Dict[str, List]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add_stat(self, category: str, key: str, value: float) -> None:
        """
        Add a float value stat to the StatsReporter.
        :param category: The highest categorization of the statistic, e.g. behavior name.
        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        """
        self.stats_dict[category][key].append(value)

    def write_stats(self, category: str, step: int) -> None:
        """
        Write out all stored statistics that fall under the category specified.
        The currently stored values will be averaged, written out as a single value,
        and the buffer cleared.
        :param category: The category which to write out the stats.
        :param step: Training step which to write these stats as.
        """
        for key in self.stats_dict[category]:
            if len(self.stats_dict[category][key]) > 0:
                stat_mean = float(np.mean(self.stats_dict[category][key]))
                for writer in self.writers:
                    writer.write_stats(category, key, stat_mean, step)
        del self.stats_dict[category]

    def write_text(self, category: str, text: str, step: int) -> None:
        """
        Write out some text.
        :param category: The highest categorization of the statistic, e.g. behavior name.
        :param text: The text to write out.
        :param step: Training step which to write these stats as.
        """
        for writer in self.writers:
            writer.write_text(category, text, step)

    def get_stats_summaries(self, category: str, key: str) -> StatsSummary:
        """
        Get the mean, std, and count of a particular statistic, since last write.
        :param category: The highest categorization of the statistic, e.g. behavior name.
        :param key: The type of statistic, e.g. Environment/Reward.
        :returns: A StatsSummary NamedTuple containing (mean, std, count).
        """
        return StatsSummary(
            mean=np.mean(self.stats_dict[category][key]),
            std=np.std(self.stats_dict[category][key]),
            num=len(self.stats_dict[category][key]),
        )


stats_reporter = StatsReporter([TensorboardWriter()])
