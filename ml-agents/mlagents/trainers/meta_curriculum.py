"""Contains the MetaCurriculum class."""

import os
from mlagents.trainers.curriculum import Curriculum
from mlagents.trainers.exception import MetaCurriculumError

import logging

logger = logging.getLogger('mlagents.trainers')


class MetaCurriculum(object):
    """A MetaCurriculum holds curriculums. Each curriculum is associated to a
    particular brain in the environment.
    """

    def __init__(self, curriculum_folder, default_reset_parameters):
        """Initializes a MetaCurriculum object.

        Args:
            curriculum_folder (str): The relative or absolute path of the
                folder which holds the curriculums for this environment.
                The folder should contain JSON files whose names are the
                brains that the curriculums belong to.
            default_reset_parameters (dict): The default reset parameters
                of the environment.
        """
        used_reset_parameters = set()
        self._brains_to_curriculums = {}

        try:
            for curriculum_filename in os.listdir(curriculum_folder):
                brain_name = curriculum_filename.split('.')[0]
                curriculum_filepath = \
                    os.path.join(curriculum_folder, curriculum_filename)
                curriculum = Curriculum(curriculum_filepath,
                                        default_reset_parameters)

                # Check if any two curriculums use the same reset params.
                if any([(parameter in curriculum.get_config().keys())
                    for parameter in used_reset_parameters]):
                    logger.warning('Two or more curriculums will '
                                'attempt to change the same reset '
                                'parameter. The result will be '
                                'non-deterministic.')

                used_reset_parameters.update(curriculum.get_config().keys())
                self._brains_to_curriculums[brain_name] = curriculum
        except NotADirectoryError:
            raise MetaCurriculumError(curriculum_folder + ' is not a '
                                      'directory. Refer to the ML-Agents '
                                      'curriculum learning docs.')


    @property
    def brains_to_curriculums(self):
        """A dict from brain_name to the brain's curriculum."""
        return self._brains_to_curriculums

    @property
    def lesson_nums(self):
        """A dict from brain name to the brain's curriculum's lesson number."""
        lesson_nums = {}
        for brain_name, curriculum in self.brains_to_curriculums.items():
            lesson_nums[brain_name] = curriculum.lesson_num

        return lesson_nums

    @lesson_nums.setter
    def lesson_nums(self, lesson_nums):
        for brain_name, lesson in lesson_nums.items():
            self.brains_to_curriculums[brain_name].lesson_num = lesson

    def increment_lessons(self, progresses):
        """Increments all the lessons of all the curriculums in this
        MetaCurriculum.

        Args:
            progresses (dict): A dict of brain name to progress.
        """
        for brain_name, progress in progresses.items():
            self.brains_to_curriculums[brain_name].increment_lesson(progress)


    def set_all_curriculums_to_lesson_num(self, lesson_num):
        """Sets all the curriculums in this meta curriculum to a specified
        lesson number.

        Args:
            lesson_num (int): The lesson number which all the curriculums will
                be set to.
        """
        for _, curriculum in self.brains_to_curriculums.items():
            curriculum.lesson_num = lesson_num


    def get_config(self):
        """Get the combined configuration of all curriculums in this
        MetaCurriculum.

        Returns:
            A dict from parameter to value.
        """
        config = {}

        for _, curriculum in self.brains_to_curriculums.items():
            curr_config = curriculum.get_config()
            config.update(curr_config)

        return config
