import os
import yaml
import math

from .exception import LessonControllerError

import logging

logger = logging.getLogger("mlagents.trainers")


class LessonController(object):
    def __init__(self, location):
        """
        Initializes a Curriculum object.
        :param location: Path to yaml file defining reset configuration
        """
        self.measure = None
        try:
            with open(location) as data_file:
                self.data = yaml.load(data_file)
        except IOError:
            raise LessonControllerError("The file {0} could not be found.".format(location))
        except UnicodeDecodeError:
            raise LessonControllerError("There was an error decoding {}".format(location))
        self.smoothing_value = 0
        self.check_keys(location)
        self.measure = self.data["measure"]
        self.thresholds = self.data["thresholds"]
        self.min_lesson_length = self.data["min_lesson_length"]
        self.max_lesson_num = len(self.thresholds)
        self._lesson_num = 0


    def check_keys(self, location):
        for key in [
            "measure",
            "thresholds",
            "min_lesson_length",
            "signal_smoothing",
        ]:
            if key not in self.data:
                raise LessonControllerError(
                    "{0} does not contain a " "{1} field.".format(location, key)
                )


    @property
    def lesson_num(self):
        return self._lesson_num

    @lesson_num.setter
    def lesson_num(self, lesson_num):
        self._lesson_num = max(0, min(lesson_num, self.max_lesson_num))

    def _lesson_ready_to_increment(self, brain_name, reward_buff_size):
        """Determines whether the curriculum of a specified brain is ready
        to attempt an increment.

        Args:
            brain_name (str): The name of the brain whose curriculum will be
                checked for readiness.
            reward_buff_size (int): The size of the reward buffer of the trainer
                that corresponds to the specified brain.

        Returns:
            Whether the curriculum of the specified brain should attempt to
            increment its lesson.
        """
        return (reward_buff_size >= self.min_lesson_length)

    def change_lesson(self, measure_val):
        """
        Increments the lesson number if threshold met

        :param measure_val: A dict of brain name to measure value.
        :return Whether the lesson was incremented.
        """
        if not self.data or not measure_val or math.isnan(measure_val):
            return False
        if self.data["signal_smoothing"]:
            measure_val = self.smoothing_value * 0.25 + 0.75 * measure_val
            self.smoothing_value = measure_val

        if (self.lesson_num < self.max_lesson_num):
            if measure_val >= self.data["thresholds"][self.lesson_num]:
                    self.lesson_num += 1
                    logger.info(
                        "Lesson changed. Now in lesson {0}".format(
                            self.lesson_num + 1,
                        )
                    )
                    return True
        return False

    def check_change_lesson(self, measure_vals, reward_buff_sizes=None):
        """Checks if the brain met the threshold defined performance. 
        Note that calling this method does not guarantee the
        lesson of a brain will increment. The lesson will
        only increment if the specified measure threshold defined in the
        param_reset_config has been reached and the minimum number of episodes in the
        lesson have been completed.

        Args:
            measure_vals (dict): A dict of brain name to measure value.
            reward_buff_sizes (dict): A dict of brain names to the size of their
                corresponding reward buffers.

        Returns:
            A dict from brain name to whether that brain's lesson was changed.
        """
        ret = {}
        if reward_buff_sizes:
            for brain_name, buff_size in reward_buff_sizes.items():
                if self._lesson_ready_to_increment(brain_name, buff_size):
                    measure_val = measure_vals[brain_name]
                    ret[brain_name] = self.change_lesson(measure_val)
        else:
            for brain_name, measure_val in measure_vals.items():
                ret[brain_name] = self.change_lesson(measure_val)
        return ret

