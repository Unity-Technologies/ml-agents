from collections import defaultdict
from enum import Enum
from typing import List, Dict, NamedTuple, Any
import numpy as np
import abc
import csv
import os
import time
import logging

from mlagents.tf_utils import tf, generate_session_config
from mlagents_envs.timers import set_gauge

logger = logging.getLogger("mlagents.trainers")


class StatsSummary(NamedTuple):
    mean: float
    std: float
    num: int

    @staticmethod
    def empty() -> "StatsSummary":
        return StatsSummary(0.0, 0.0, 0)


class StatsPropertyType(Enum):
    HYPERPARAMETERS = "hyperparameters"
    SELF_PLAY = "selfplay"
    SELF_PLAY_TEAM = "selfplayteam"


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

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        """
        Add a generic property to the StatsWriter. This could be e.g. a Dict of hyperparameters,
        a max step count, a trainer type, etc. Note that not all StatsWriters need to be compatible
        with all types of properties. For instance, a TB writer doesn't need a max step, nor should
        we write hyperparameters to the CSV.
        :param category: The category that the property belongs to.
        :param type: The type of property.
        :param value: The property itself.
        """
        pass


class GaugeWriter(StatsWriter):
    """
    Write all stats that we recieve to the timer gauges, so we can track them offline easily
    """

    @staticmethod
    def sanitize_string(s: str) -> str:
        """
        Clean up special characters in the category and value names.
        """
        return s.replace("/", ".").replace(" ", "")

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for val, stats_summary in values.items():
            set_gauge(
                GaugeWriter.sanitize_string(f"{category}.{val}.mean"),
                float(stats_summary.mean),
            )


class ConsoleWriter(StatsWriter):
    def __init__(self):
        self.training_start_time = time.time()
        # If self-play, we want to print ELO as well as reward
        self.self_play = False
        self.self_play_team = -1

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        is_training = "Not Training."
        if "Is Training" in values:
            stats_summary = stats_summary = values["Is Training"]
            if stats_summary.mean > 0.0:
                is_training = "Training."

        if "Environment/Cumulative Reward" in values:
            stats_summary = values["Environment/Cumulative Reward"]
            logger.info(
                "{}: Step: {}. "
                "Time Elapsed: {:0.3f} s "
                "Mean "
                "Reward: {:0.3f}"
                ". Std of Reward: {:0.3f}. {}".format(
                    category,
                    step,
                    time.time() - self.training_start_time,
                    stats_summary.mean,
                    stats_summary.std,
                    is_training,
                )
            )
            if self.self_play and "Self-play/ELO" in values:
                elo_stats = values["Self-play/ELO"]
                mean_opponent_elo = values["Self-play/Mean Opponent ELO"]
                std_opponent_elo = values["Self-play/Std Opponent ELO"]
                logger.info(
                    "{} Team {}: ELO: {:0.3f}. "
                    "Mean Opponent ELO: {:0.3f}. "
                    "Std Opponent ELO: {:0.3f}. ".format(
                        category,
                        self.self_play_team,
                        elo_stats.mean,
                        mean_opponent_elo.mean,
                        std_opponent_elo.mean,
                    )
                )
        else:
            logger.info(
                "{}: Step: {}. No episode was completed since last summary. {}".format(
                    category, step, is_training
                )
            )

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            logger.info(
                """Hyperparameters for behavior name {0}: \n{1}""".format(
                    category, self._dict_to_str(value, 0)
                )
            )
        elif property_type == StatsPropertyType.SELF_PLAY:
            assert isinstance(value, bool)
            self.self_play = value
        elif property_type == StatsPropertyType.SELF_PLAY_TEAM:
            assert isinstance(value, int)
            self.self_play_team = value

    def _dict_to_str(self, param_dict: Dict[str, Any], num_tabs: int) -> str:
        """
        Takes a parameter dictionary and converts it to a human-readable string.
        Recurses if there are multiple levels of dict. Used to print out hyperparameters.
        param: param_dict: A Dictionary of key, value parameters.
        return: A string version of this dictionary.
        """
        if not isinstance(param_dict, dict):
            return str(param_dict)
        else:
            append_newline = "\n" if num_tabs > 0 else ""
            return append_newline + "\n".join(
                [
                    "\t"
                    + "  " * num_tabs
                    + "{0}:\t{1}".format(
                        x, self._dict_to_str(param_dict[x], num_tabs + 1)
                    )
                    for x in param_dict
                ]
            )


class TensorboardWriter(StatsWriter):
    def __init__(self, base_dir: str):
        """
        A StatsWriter that writes to a Tensorboard summary.
        :param base_dir: The directory within which to place all the summaries. Tensorboard files will be written to a
        {base_dir}/{category} directory.
        """
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

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            assert isinstance(value, dict)
            text = self._dict_to_tensorboard("Hyperparameters", value)
            self._maybe_create_summary_writer(category)
            self.summary_writers[category].add_summary(text, 0)

    def _dict_to_tensorboard(self, name: str, input_dict: Dict[str, Any]) -> str:
        """
        Convert a dict to a Tensorboard-encoded string.
        :param name: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            with tf.Session(config=generate_session_config()) as sess:
                s_op = tf.summary.text(
                    name,
                    tf.convert_to_tensor(
                        ([[str(x), str(input_dict[x])] for x in input_dict])
                    ),
                )
                s = sess.run(s_op)
                return s
        except Exception:
            logger.warning("Could not write text summary for Tensorboard.")
            return ""


class CSVWriter(StatsWriter):
    def __init__(self, base_dir: str, required_fields: List[str] = None):
        """
        A StatsWriter that writes to a Tensorboard summary.
        :param base_dir: The directory within which to place the CSV file, which will be {base_dir}/{category}.csv.
        :param required_fields: If provided, the CSV writer won't write until these fields have statistics to write for
        them.
        """
        # We need to keep track of the fields in the CSV, as all rows need the same fields.
        self.csv_fields: Dict[str, List[str]] = {}
        self.required_fields = required_fields if required_fields else []
        self.base_dir: str = base_dir

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        if self._maybe_create_csv_file(category, list(values.keys())):
            row = [str(step)]
            # Only record the stats that showed up in the first valid row
            for key in self.csv_fields[category]:
                _val = values.get(key, None)
                row.append(str(_val.mean) if _val else "None")
            with open(self._get_filepath(category), "a") as file:
                writer = csv.writer(file)
                writer.writerow(row)

    def _maybe_create_csv_file(self, category: str, keys: List[str]) -> bool:
        """
        If no CSV file exists and the keys have the required values,
        make the CSV file and write hte title row.
        Returns True if there is now (or already is) a valid CSV file.
        """
        if category not in self.csv_fields:
            summary_dir = self.base_dir
            os.makedirs(summary_dir, exist_ok=True)
            # Only store if the row contains the required fields
            if all(item in keys for item in self.required_fields):
                self.csv_fields[category] = keys
                with open(self._get_filepath(category), "w") as file:
                    title_row = ["Steps"]
                    title_row.extend(keys)
                    writer = csv.writer(file)
                    writer.writerow(title_row)
                return True
            return False
        return True

    def _get_filepath(self, category: str) -> str:
        file_dir = os.path.join(self.base_dir, category + ".csv")
        return file_dir


class StatsReporter:
    writers: List[StatsWriter] = []
    stats_dict: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    def __init__(self, category: str):
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

    def add_property(self, property_type: StatsPropertyType, value: Any) -> None:
        """
        Add a generic property to the StatsReporter. This could be e.g. a Dict of hyperparameters,
        a max step count, a trainer type, etc. Note that not all StatsWriters need to be compatible
        with all types of properties. For instance, a TB writer doesn't need a max step, nor should
        we write hyperparameters to the CSV.
        :param key: The type of property.
        :param value: The property itself.
        """
        for writer in StatsReporter.writers:
            writer.add_property(self.category, property_type, value)

    def add_stat(self, key: str, value: float) -> None:
        """
        Add a float value stat to the StatsReporter.
        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        """
        StatsReporter.stats_dict[self.category][key].append(value)

    def set_stat(self, key: str, value: float) -> None:
        """
        Sets a stat value to a float. This is for values that we don't want to average, and just
        want the latest.
        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        """
        StatsReporter.stats_dict[self.category][key] = [value]

    def write_stats(self, step: int) -> None:
        """
        Write out all stored statistics that fall under the category specified.
        The currently stored values will be averaged, written out as a single value,
        and the buffer cleared.
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

    def get_stats_summaries(self, key: str) -> StatsSummary:
        """
        Get the mean, std, and count of a particular statistic, since last write.
        :param key: The type of statistic, e.g. Environment/Reward.
        :returns: A StatsSummary NamedTuple containing (mean, std, count).
        """
        if len(StatsReporter.stats_dict[self.category][key]) > 0:
            return StatsSummary(
                mean=np.mean(StatsReporter.stats_dict[self.category][key]),
                std=np.std(StatsReporter.stats_dict[self.category][key]),
                num=len(StatsReporter.stats_dict[self.category][key]),
            )
        return StatsSummary.empty()
