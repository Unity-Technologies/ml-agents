from collections import defaultdict
from enum import Enum
from typing import List, Dict, NamedTuple, Any, Optional
import numpy as np
import abc
import os
import time
from threading import RLock

from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import set_gauge
from mlagents.tf_utils import tf, generate_session_config
from mlagents.tf_utils.globals import get_rank


logger = get_logger(__name__)


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
        with all types of properties. For instance, a TB writer doesn't need a max step.
        :param category: The category that the property belongs to.
        :param type: The type of property.
        :param value: The property itself.
        """
        pass


class GaugeWriter(StatsWriter):
    """
    Write all stats that we receive to the timer gauges, so we can track them offline easily
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
        self.rank = get_rank()

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        is_training = "Not Training."
        if "Is Training" in values:
            stats_summary = values["Is Training"]
            if stats_summary.mean > 0.0:
                is_training = "Training."

        elapsed_time = time.time() - self.training_start_time
        log_info: List[str] = [category]
        log_info.append(f"Step: {step}")
        log_info.append(f"Time Elapsed: {elapsed_time:0.3f} s")
        if "Environment/Cumulative Reward" in values:
            stats_summary = values["Environment/Cumulative Reward"]
            if self.rank is not None:
                log_info.append(f"Rank: {self.rank}")

            log_info.append(f"Mean Reward: {stats_summary.mean:0.3f}")
            log_info.append(f"Std of Reward: {stats_summary.std:0.3f}")
            log_info.append(is_training)

            if self.self_play and "Self-play/ELO" in values:
                elo_stats = values["Self-play/ELO"]
                log_info.append(f"ELO: {elo_stats.mean:0.3f}")
        else:
            log_info.append("No episode was completed since last summary")
            log_info.append(is_training)
        logger.info(". ".join(log_info))

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            logger.info(
                """Hyperparameters for behavior name {}: \n{}""".format(
                    category, self._dict_to_str(value, 0)
                )
            )
        elif property_type == StatsPropertyType.SELF_PLAY:
            assert isinstance(value, bool)
            self.self_play = value

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
                    + "{}:\t{}".format(
                        x, self._dict_to_str(param_dict[x], num_tabs + 1)
                    )
                    for x in param_dict
                ]
            )


class TensorboardWriter(StatsWriter):
    def __init__(self, base_dir: str, clear_past_data: bool = False):
        """
        A StatsWriter that writes to a Tensorboard summary.
        :param base_dir: The directory within which to place all the summaries. Tensorboard files will be written to a
        {base_dir}/{category} directory.
        :param clear_past_data: Whether or not to clean up existing Tensorboard files associated with the base_dir and
            category.
        """
        self.summary_writers: Dict[str, tf.summary.FileWriter] = {}
        self.base_dir: str = base_dir
        self._clear_past_data = clear_past_data

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self._maybe_create_summary_writer(category)
        for key, value in values.items():
            summary = tf.Summary()
            summary.value.add(tag=f"{key}", simple_value=value.mean)
            self.summary_writers[category].add_summary(summary, step)
            self.summary_writers[category].flush()

    def _maybe_create_summary_writer(self, category: str) -> None:
        if category not in self.summary_writers:
            filewriter_dir = "{basedir}/{category}".format(
                basedir=self.base_dir, category=category
            )
            os.makedirs(filewriter_dir, exist_ok=True)
            if self._clear_past_data:
                self._delete_all_events_files(filewriter_dir)
            self.summary_writers[category] = tf.summary.FileWriter(filewriter_dir)

    def _delete_all_events_files(self, directory_name: str) -> None:
        for file_name in os.listdir(directory_name):
            if file_name.startswith("events.out"):
                logger.warning(
                    f"{file_name} was left over from a previous run. Deleting."
                )
                full_fname = os.path.join(directory_name, file_name)
                try:
                    os.remove(full_fname)
                except OSError:
                    logger.warning(
                        "{} was left over from a previous run and "
                        "not deleted.".format(full_fname)
                    )

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            assert isinstance(value, dict)
            summary = self._dict_to_tensorboard("Hyperparameters", value)
            self._maybe_create_summary_writer(category)
            if summary is not None:
                self.summary_writers[category].add_summary(summary, 0)

    def _dict_to_tensorboard(
        self, name: str, input_dict: Dict[str, Any]
    ) -> Optional[bytes]:
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
                        [[str(x), str(input_dict[x])] for x in input_dict]
                    ),
                )
                s = sess.run(s_op)
                return s
        except Exception:
            logger.warning(
                f"Could not write {name} summary for Tensorboard: {input_dict}"
            )
            return None


class StatsReporter:
    writers: List[StatsWriter] = []
    stats_dict: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    lock = RLock()

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
        with StatsReporter.lock:
            StatsReporter.writers.append(writer)

    def add_property(self, property_type: StatsPropertyType, value: Any) -> None:
        """
        Add a generic property to the StatsReporter. This could be e.g. a Dict of hyperparameters,
        a max step count, a trainer type, etc. Note that not all StatsWriters need to be compatible
        with all types of properties. For instance, a TB writer doesn't need a max step.
        :param key: The type of property.
        :param value: The property itself.
        """
        with StatsReporter.lock:
            for writer in StatsReporter.writers:
                writer.add_property(self.category, property_type, value)

    def add_stat(self, key: str, value: float) -> None:
        """
        Add a float value stat to the StatsReporter.
        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        """
        with StatsReporter.lock:
            StatsReporter.stats_dict[self.category][key].append(value)

    def set_stat(self, key: str, value: float) -> None:
        """
        Sets a stat value to a float. This is for values that we don't want to average, and just
        want the latest.
        :param key: The type of statistic, e.g. Environment/Reward.
        :param value: the value of the statistic.
        """
        with StatsReporter.lock:
            StatsReporter.stats_dict[self.category][key] = [value]

    def write_stats(self, step: int) -> None:
        """
        Write out all stored statistics that fall under the category specified.
        The currently stored values will be averaged, written out as a single value,
        and the buffer cleared.
        :param step: Training step which to write these stats as.
        """
        with StatsReporter.lock:
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
