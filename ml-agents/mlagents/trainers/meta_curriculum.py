"""Contains the MetaCurriculum class."""

from typing import Dict, Set
from mlagents.trainers.curriculum import Curriculum
from mlagents.trainers.settings import CurriculumSettings
from mlagents.trainers.training_status import GlobalTrainingStatus, StatusType

from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


class MetaCurriculum:
    """A MetaCurriculum holds curricula. Each curriculum is associated to a
    particular brain in the environment.
    """

    def __init__(self, curriculum_configs: Dict[str, CurriculumSettings]):
        """Initializes a MetaCurriculum object.

        :param curriculum_folder: Dictionary of brain_name to the
          Curriculum for each brain.
        """
        self._brains_to_curricula: Dict[str, Curriculum] = {}
        used_reset_parameters: Set[str] = set()
        for brain_name, curriculum_settings in curriculum_configs.items():
            self._brains_to_curricula[brain_name] = Curriculum(
                brain_name, curriculum_settings
            )
            config_keys: Set[str] = set(
                self._brains_to_curricula[brain_name].get_config().keys()
            )

            # Check if any two curricula use the same reset params.
            if config_keys & used_reset_parameters:
                logger.warning(
                    "Two or more curricula will "
                    "attempt to change the same reset "
                    "parameter. The result will be "
                    "non-deterministic."
                )

            used_reset_parameters.update(config_keys)

    @property
    def brains_to_curricula(self):
        """A dict from brain_name to the brain's curriculum."""
        return self._brains_to_curricula

    @property
    def lesson_nums(self):
        """A dict from brain name to the brain's curriculum's lesson number."""
        lesson_nums = {}
        for brain_name, curriculum in self.brains_to_curricula.items():
            lesson_nums[brain_name] = curriculum.lesson_num

        return lesson_nums

    @lesson_nums.setter
    def lesson_nums(self, lesson_nums):
        for brain_name, lesson in lesson_nums.items():
            self.brains_to_curricula[brain_name].lesson_num = lesson

    def _lesson_ready_to_increment(
        self, brain_name: str, reward_buff_size: int
    ) -> bool:
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
        if brain_name not in self.brains_to_curricula:
            return False

        return reward_buff_size >= (
            self.brains_to_curricula[brain_name].min_lesson_length
        )

    def increment_lessons(self, measure_vals, reward_buff_sizes=None):
        """Attempts to increments all the lessons of all the curricula in this
        MetaCurriculum. Note that calling this method does not guarantee the
        lesson of a curriculum will increment. The lesson of a curriculum will
        only increment if the specified measure threshold defined in the
        curriculum has been reached and the minimum number of episodes in the
        lesson have been completed.

        Args:
            measure_vals (dict): A dict of brain name to measure value.
            reward_buff_sizes (dict): A dict of brain names to the size of their
                corresponding reward buffers.

        Returns:
            A dict from brain name to whether that brain's lesson number was
            incremented.
        """
        ret = {}
        if reward_buff_sizes:
            for brain_name, buff_size in reward_buff_sizes.items():
                if self._lesson_ready_to_increment(brain_name, buff_size):
                    measure_val = measure_vals[brain_name]
                    ret[brain_name] = self.brains_to_curricula[
                        brain_name
                    ].increment_lesson(measure_val)
        else:
            for brain_name, measure_val in measure_vals.items():
                ret[brain_name] = self.brains_to_curricula[brain_name].increment_lesson(
                    measure_val
                )
        return ret

    def try_restore_all_curriculum(self):
        """
        Tries to restore all the curriculums to what is saved in training_status.json
        """

        for brain_name, curriculum in self.brains_to_curricula.items():
            lesson_num = GlobalTrainingStatus.get_parameter_state(
                brain_name, StatusType.LESSON_NUM
            )
            if lesson_num is not None:
                logger.info(
                    f"Resuming curriculum for {brain_name} at lesson {lesson_num}."
                )
                curriculum.lesson_num = lesson_num
            else:
                curriculum.lesson_num = 0

    def get_config(self):
        """Get the combined configuration of all curricula in this
        MetaCurriculum.

        :return: A dict from parameter to value.
        """
        config = {}

        for _, curriculum in self.brains_to_curricula.items():
            curr_config = curriculum.get_config()
            config.update(curr_config)

        return config
