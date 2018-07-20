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
            for curriculum_filename in os.listdir(curriculum_folder):
                brain_name = curriculum_filename.split('.')[0]
                curriculum_filepath = os.path.join(curriculum_folder, curriculum_filename)
                self._brains_to_curriculums[brain_name] = Curriculum(curriculum_filepath, default_reset_parameters)

    @property
    def brains_to_curriculums(self):
        return self._brains_to_curriculums

    @property
    def lesson_nums(self):
        lesson_nums = {}
        for brain_name, curriculum in self.brains_to_curriculums:
            lesson_nums[brain_name] = curriculum.lesson_num

        return lesson_nums

    @lesson_nums.setter
    def lesson_nums(self, lesson_nums):
        for brain_name, lesson in lesson_nums.items():
            self.brains_to_curriculums[brain_name].lesson_num = lesson

    def increment_lessons(self, progresses):
        for brain_name, progress in progresses.items():
            self.brains_to_curriculums[brain_name].increment_lesson(progress)


    def set_all_curriculums_to_lesson_num(self, lesson_num):
        for _, curriculum in self.brains_to_curriculums.items():
            curriculum.lesson_num = lesson_num


    def get_config(self):
        config = {}

        for _, curriculum in self.brains_to_curriculums.items():
            parameters = curriculum.data["parameters"]
            for key in parameters:
                config[key] = parameters[key][curriculum.lesson_num]

        return config
