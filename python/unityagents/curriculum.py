import json
import numpy as np

from .exception import UnityEnvironmentException

class Curriculum(object):
    def __init__(self, location, default_reset_parameters):
        self.lesson_number = 0
        self.lesson_length = 0
        self.measure_type = None
        if location == None:
            self.data = None
        else:
            try:
                with open(location) as data_file:
                    self.data = json.load(data_file)
            except:
                raise UnityEnvironmentException(
                        "The file {0} could not be found.".format(location))
            parameters = self.data["parameters"]
            self.measure_type = self.data["measure"]
            self.max_lesson_number = len(self.data['thresholds'])
            self.smoothing_value = 0
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


    @property
    def measure(self):
        return self.measure_type

    def get_lesson_number(self):
        return self.lesson_number

    def set_lesson_number(self, value):
        self.lesson_length = 0
        self.lesson_number = max(0,min(value,self.max_lesson_number))

    def get_lesson(self, progress):
        if (self.data == None ) or (progress == None):
            return {}
        if self.data["signal_smoothing"]:
            progress = self.smoothing_value*0.9 + 0.1*progress
            self.smoothing_value = progress
        self.lesson_length += 1
        if self.lesson_number < self.max_lesson_number:
            if ((progress > self.data['thresholds'][self.lesson_number])
                and (self.lesson_length > self.data['min_lesson_length'])):
                self.lesson_length = 0
                self.lesson_number += 1
        config = {}
        parameters = self.data["parameters"]
        for key in parameters:
            config[key] = parameters[key][self.lesson_number]
        return config




