import os
import json
import math
from typing import Dict, Any, TextIO

from .exception import CurriculumConfigError, CurriculumLoadingError

import logging

logger = logging.getLogger("mlagents.trainers")


class Curriculum(object):
    def __init__(self, location):
        """
        Initializes a Curriculum object.
        :param location: Path to JSON defining curriculum.
        """
        self.max_lesson_num = 0
        self.measure = None
        self._lesson_num = 0
        # The name of the brain should be the basename of the file without the
        # extension.
        self._brain_name = os.path.basename(location).split(".")[0]
        self.data = Curriculum.load_curriculum_file(location)

        self.smoothing_value = 0
        for key in [
            "parameters",
            "measure",
            "thresholds",
            "min_lesson_length",
            "signal_smoothing",
        ]:
            if key not in self.data:
                raise CurriculumConfigError(
                    "{0} does not contain a " "{1} field.".format(location, key)
                )
        self.smoothing_value = 0
        self.measure = self.data["measure"]
        self.min_lesson_length = self.data["min_lesson_length"]
        self.max_lesson_num = len(self.data["thresholds"])

        parameters = self.data["parameters"]
        for key in parameters:
            if len(parameters[key]) != self.max_lesson_num + 1:
                raise CurriculumConfigError(
                    "The parameter {0} in Curriculum {1} must have {2} values "
                    "but {3} were found".format(
                        key, location, self.max_lesson_num + 1, len(parameters[key])
                    )
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
        if not self.data or not measure_val or math.isnan(measure_val):
            return False
        if self.data["signal_smoothing"]:
            measure_val = self.smoothing_value * 0.25 + 0.75 * measure_val
            self.smoothing_value = measure_val
        if self.lesson_num < self.max_lesson_num:
            if measure_val > self.data["thresholds"][self.lesson_num]:
                self.lesson_num += 1
                config = {}
                parameters = self.data["parameters"]
                for key in parameters:
                    config[key] = parameters[key][self.lesson_num]
                logger.info(
                    "{0} lesson changed. Now in lesson {1}: {2}".format(
                        self._brain_name,
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
        if not self.data:
            return {}
        if lesson is None:
            lesson = self.lesson_num
        lesson = max(0, min(lesson, self.max_lesson_num))
        config = {}
        parameters = self.data["parameters"]
        for key in parameters:
            config[key] = parameters[key][lesson]
        return config

    @staticmethod
    def load_curriculum_file(location: str) -> None:
        try:
            with open(location) as data_file:
                return Curriculum._load_curriculum(data_file)
        except IOError:
            raise CurriculumLoadingError(
                "The file {0} could not be found.".format(location)
            )
        except UnicodeDecodeError:
            raise CurriculumLoadingError(
                "There was an error decoding {}".format(location)
            )

    @staticmethod
    def _load_curriculum(fp: TextIO) -> None:
        try:
            return json.load(fp)
        except json.decoder.JSONDecodeError as e:
            raise CurriculumLoadingError(
                "Error parsing JSON file. Please check for formatting errors. "
                "A tool such as https://jsonlint.com/ can be helpful with this."
            ) from e
