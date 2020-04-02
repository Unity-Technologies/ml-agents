import json
import math
from typing import Dict, Any, TextIO

from .exception import CurriculumConfigError, CurriculumLoadingError

from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


class Curriculum:
    def __init__(self, brain_name: str, config: Dict):
        """
        Initializes a Curriculum object.
        :param brain_name: Name of the brain this Curriculum is associated with
        :param config: Dictionary of fields needed to configure the Curriculum
        """
        self.max_lesson_num = 0
        self.measure = None
        self._lesson_num = 0
        self.brain_name = brain_name
        self.config = config

        self.smoothing_value = 0.0
        for key in [
            "parameters",
            "measure",
            "thresholds",
            "min_lesson_length",
            "signal_smoothing",
        ]:
            if key not in self.config:
                raise CurriculumConfigError(
                    f"{brain_name} curriculum config does not contain a {key} field."
                )
        self.smoothing_value = 0
        self.measure = self.config["measure"]
        self.min_lesson_length = self.config["min_lesson_length"]
        self.max_lesson_num = len(self.config["thresholds"])

        parameters = self.config["parameters"]
        for key in parameters:
            if len(parameters[key]) != self.max_lesson_num + 1:
                raise CurriculumConfigError(
                    f"The parameter {key} in {brain_name}'s curriculum must have {self.max_lesson_num + 1} values "
                    f"but {len(parameters[key])} were found"
                )

    @property
    def lesson_num(self) -> int:
        return self._lesson_num

    @lesson_num.setter
    def lesson_num(self, lesson_num: int) -> None:
        self._lesson_num = max(0, min(lesson_num, self.max_lesson_num))

    def increment_lesson(self, measure_val: float) -> bool:
        """
        Increments the lesson number depending on the progress given.
        :param measure_val: Measure of progress (either reward or percentage
               steps completed).
        :return Whether the lesson was incremented.
        """
        if not self.config or not measure_val or math.isnan(measure_val):
            return False
        if self.config["signal_smoothing"]:
            measure_val = self.smoothing_value * 0.25 + 0.75 * measure_val
            self.smoothing_value = measure_val
        if self.lesson_num < self.max_lesson_num:
            if measure_val > self.config["thresholds"][self.lesson_num]:
                self.lesson_num += 1
                config = {}
                parameters = self.config["parameters"]
                for key in parameters:
                    config[key] = parameters[key][self.lesson_num]
                logger.info(
                    "{0} lesson changed. Now in lesson {1}: {2}".format(
                        self.brain_name,
                        self.lesson_num,
                        ", ".join([str(x) + " -> " + str(config[x]) for x in config]),
                    )
                )
                return True
        return False

    def get_config(self, lesson: int = None) -> Dict[str, Any]:
        """
        Returns reset parameters which correspond to the lesson.
        :param lesson: The lesson you want to get the config of. If None, the
               current lesson is returned.
        :return: The configuration of the reset parameters.
        """
        if not self.config:
            return {}
        if lesson is None:
            lesson = self.lesson_num
        lesson = max(0, min(lesson, self.max_lesson_num))
        config = {}
        parameters = self.config["parameters"]
        for key in parameters:
            config[key] = parameters[key][lesson]
        return config

    @staticmethod
    def load_curriculum_file(config_path: str) -> Dict:
        try:
            with open(config_path) as data_file:
                return Curriculum._load_curriculum(data_file)
        except IOError:
            raise CurriculumLoadingError(
                "The file {0} could not be found.".format(config_path)
            )
        except UnicodeDecodeError:
            raise CurriculumLoadingError(
                "There was an error decoding {}".format(config_path)
            )

    @staticmethod
    def _load_curriculum(fp: TextIO) -> Dict:
        try:
            return json.load(fp)
        except json.decoder.JSONDecodeError as e:
            raise CurriculumLoadingError(
                "Error parsing JSON file. Please check for formatting errors. "
                "A tool such as https://jsonlint.com/ can be helpful with this."
            ) from e
