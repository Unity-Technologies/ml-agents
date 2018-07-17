"""
A School holds many curriculums. The School tracks which brains are following which curriculums.
"""

import os
from unitytrainers import Curriculum

class School:
    def __init__(self, curriculum_folder, default_reset_parameters):
        """
        Initializes a School object.
        """
        if curriculum_folder is None:
            self._brains_to_curriculums = None
        else:
            self._brains_to_curriculums = {}
            for location in os.listdir(curriculum_folder):
                brain_name = location.split('.')[0]
                self._brains_to_curriculums[brain_name] = Curriculum(location, default_reset_parameters)

    @property
    def brains_to_curriculums(self):
        return self._brains_to_curriculums


    def increment_lessons(self, progresses):
        for brain_name, progress in progresses.items():
            self.brains_to_curriculums[brain_name].increment_lesson(progress)


    def set_lesson_nums(self, lesson_nums):
        for brain_name, lesson in lesson_nums.items():
            self.brains_to_curriculums[brain_name].lesson_number = lesson


    def get_config(self):
        config = {}

        for _, curriculum in self.brains_to_curriculums.items():
            parameters = curriculum.data["parameters"]
            for key in parameters:
                config[key] = parameters[key][curriculum.lesson_number]

        return config
