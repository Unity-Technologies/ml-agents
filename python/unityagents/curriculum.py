import json

from .exception import UnityEnvironmentException

import logging

logger = logging.getLogger("unityagents")


class Curriculum(object):
    def __init__(self, location, default_reset_parameters):
        """
        Initializes a Curriculum object.
        :param location: Path to JSON defining curriculum.
        :param default_reset_parameters: Set of reset parameters for environment.
        """
        self.lesson_length = 0
        self.max_lesson_number = 0
        self.measure_type = None
        if location is None:
            self.data = None
        else:
            try:
                with open(location) as data_file:
                    self.data = json.load(data_file)
            except IOError:
                raise UnityEnvironmentException(
                    "The file {0} could not be found.".format(location))
            except UnicodeDecodeError:
                raise UnityEnvironmentException("There was an error decoding {}".format(location))
            self.smoothing_value = 0
            for key in ['parameters', 'measure', 'thresholds',
                        'min_lesson_length', 'signal_smoothing']:
                if key not in self.data:
                    raise UnityEnvironmentException("{0} does not contain a "
                                                    "{1} field.".format(location, key))
            parameters = self.data['parameters']
            self.measure_type = self.data['measure']
            self.max_lesson_number = len(self.data['thresholds'])
            for key in parameters:
                if key not in default_reset_parameters:
                    raise UnityEnvironmentException(
                        "The parameter {0} in Curriculum {1} is not present in "
                        "the Environment".format(key, location))
            for key in parameters:
                if len(parameters[key]) != self.max_lesson_number + 1:
                    raise UnityEnvironmentException(
                        "The parameter {0} in Curriculum {1} must have {2} values "
                        "but {3} were found".format(key, location,
                                                    self.max_lesson_number + 1, len(parameters[key])))
        self.set_lesson_number(0)

    @property
    def measure(self):
        return self.measure_type

    @property
    def get_lesson_number(self):
        return self.lesson_number

    def set_lesson_number(self, value):
        self.lesson_length = 0
        self.lesson_number = max(0, min(value, self.max_lesson_number))

    def increment_lesson(self, progress):
        """
        Increments the lesson number depending on the progree given.
        :param progress: Measure of progress (either reward or percentage steps completed).
        """
        if self.data is None or progress is None:
            return
        if self.data["signal_smoothing"]:
            progress = self.smoothing_value * 0.25 + 0.75 * progress
            self.smoothing_value = progress
        self.lesson_length += 1
        if self.lesson_number < self.max_lesson_number:
            if ((progress > self.data['thresholds'][self.lesson_number]) and
                    (self.lesson_length > self.data['min_lesson_length'])):
                self.lesson_length = 0
                self.lesson_number += 1
                config = {}
                parameters = self.data["parameters"]
                for key in parameters:
                    config[key] = parameters[key][self.lesson_number]
                logger.info("\nLesson changed. Now in Lesson {0} : \t{1}"
                            .format(self.lesson_number,
                                    ', '.join([str(x) + ' -> ' + str(config[x]) for x in config])))

    def get_config(self, lesson=None):
        """
        Returns reset parameters which correspond to the lesson.
        :param lesson: The lesson you want to get the config of. If None, the current lesson is returned.
        :return: The configuration of the reset parameters.
        """
        if self.data is None:
            return {}
        if lesson is None:
            lesson = self.lesson_number
        lesson = max(0, min(lesson, self.max_lesson_number))
        config = {}
        parameters = self.data["parameters"]
        for key in parameters:
            config[key] = parameters[key][lesson]
        return config
