from collections import defaultdict
from typing import List, Dict, NamedTuple
import numpy as np
import abc
import csv
import os

from mlagents.tf_utils import tf


class StatsSummary(NamedTuple):
    mean: float
    std: float
    num: int


class StatsWriter(abc.ABC):
    """
    A StatsWriter abstract class. A StatsWriter takes in a category, key, scalar value, and step
    and writes it out by some method.
    """

    @abc.abstractmethod
    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        pass

    @abc.abstractmethod
    def write_text(self, category: str, text: str, step: int) -> None:
        pass


class TensorboardWriter(StatsWriter):
    def __init__(self, base_dir: str):
        self.summary_writers: Dict[str, tf.summary.FileWriter] = {}
        self.base_dir: str = base_dir

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self._maybe_create_summary_writer(category)
        for key, value in values.items():
            summary = tf.Summary()
            summary.value.add(tag="{}".format(key), simple_value=value.mean)
            self.summary_writers[category].add_summary(summary, step)
            self.summary_writers[category].flush()

    def _maybe_create_summary_writer(self, category: str) -> None:
        if category not in self.summary_writers:
            filewriter_dir = "{basedir}/{category}".format(
                basedir=self.base_dir, category=category
            )
            os.makedirs(filewriter_dir, exist_ok=True)
            self.summary_writers[category] = tf.summary.FileWriter(filewriter_dir)

    def write_text(self, category: str, text: str, step: int) -> None:
        self._maybe_create_summary_writer(category)
        self.summary_writers[category].add_summary(text, step)


class CSVWriter(StatsWriter):
    def __init__(self, base_dir: str):
        # We need to keep track of the fields in the CSV, as all rows need the same fields.
        self.csv_fields: Dict[str, List[str]] = {}
        self.base_dir: str = base_dir

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self._maybe_create_csv_file(category, list(values.keys()))
        row = [str(step)]
        # Only record the stats that showed up in the first row.
        for key in self.csv_fields[category]:
            _val = values.get(key, None)
            row.append(str(_val.mean) if _val else "None")
        with open(self._get_filepath(category), "a") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def _maybe_create_csv_file(self, category: str, keys: List[str]) -> None:
        if category not in self.csv_fields:
            summary_dir = self.base_dir
            os.makedirs(summary_dir, exist_ok=True)
            self.csv_fields[category] = keys
            with open(self._get_filepath(category), "w") as file:
                title_row = ["Steps"]
                title_row.extend(keys)
                writer = csv.writer(file)
                writer.writerow(title_row)
                self.csv_fields[category] = keys

    def _get_filepath(self, category: str) -> str:
        file_dir = os.path.join(self.base_dir, category + ".csv")
        return file_dir

    def write_text(self, category: str, text: str, step: int) -> None:
        pass


class StatsReporter:
    writers: List[StatsWriter] = []
    stats_dict: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    def __init__(self, category):
        """
        Generic StatsReporter. A category is the broadest type of storage (would
        correspond the run name and trainer name, e.g. 3DBalltest_3DBall. A key is the
        type of stat it is (e.g. Environment/Reward). Finally the Value is the float value
        attached to this stat.
        """
        self.category: str = category

    @staticmethod
    def add_writer(writer: StatsWriter) -> None:
        StatsReporter.writers.append(writer)

    def add_stat(self, key: str, value: float) -> None:
        """
        Add a float value stat to the StatsReporter.
        :param category: The highest categorization of the statistic, e.g. behavior name.
        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        """
        StatsReporter.stats_dict[self.category][key].append(value)

    def write_stats(self, step: int) -> None:
        """
        Write out all stored statistics that fall under the category specified.
        The currently stored values will be averaged, written out as a single value,
        and the buffer cleared.
        :param category: The category which to write out the stats.
        :param step: Training step which to write these stats as.
        """
        values: Dict[str, StatsSummary] = {}
        for key in StatsReporter.stats_dict[self.category]:
            if len(StatsReporter.stats_dict[self.category][key]) > 0:
                stat_summary = self.get_stats_summaries(key)
                values[key] = stat_summary
        for writer in StatsReporter.writers:
            writer.write_stats(self.category, values, step)
        del StatsReporter.stats_dict[self.category]

    def write_text(self, text: str, step: int) -> None:
        """
        Write out some text.
        :param category: The highest categorization of the statistic, e.g. behavior name.
        :param text: The text to write out.
        :param step: Training step which to write these stats as.
        """
        for writer in StatsReporter.writers:
            writer.write_text(self.category, text, step)

    def get_stats_summaries(self, key: str) -> StatsSummary:
        """
        Get the mean, std, and count of a particular statistic, since last write.
        :param category: The highest categorization of the statistic, e.g. behavior name.
        :param key: The type of statistic, e.g. Environment/Reward.
        :returns: A StatsSummary NamedTuple containing (mean, std, count).
        """
        return StatsSummary(
            mean=np.mean(StatsReporter.stats_dict[self.category][key]),
            std=np.std(StatsReporter.stats_dict[self.category][key]),
            num=len(StatsReporter.stats_dict[self.category][key]),
        )
