from typing import Dict, List, Tuple, Optional
from mlagents.trainers.settings import (
    EnvironmentParameterSettings,
    ParameterRandomizationSettings,
)
from collections import defaultdict
from mlagents.trainers.training_status import GlobalTrainingStatus, StatusType

from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


class EnvironmentParameterManager:
    def __init__(
        self,
        settings: Optional[Dict[str, EnvironmentParameterSettings]] = None,
        run_seed: int = -1,
        restore: bool = False,
    ):
        """
        EnvironmentParameterManager manages all the environment parameters of a training
        session. It determines when parameters should change and gives access to the
        current sampler of each parameter.
        :param settings: A dictionary from environment parameter to
        EnvironmentParameterSettings.
        :param run_seed: When the seed is not provided for an environment parameter,
        this seed will be used instead.
        :param restore: If true, the EnvironmentParameterManager will use the
        GlobalTrainingStatus to try and reload the lesson status of each environment
        parameter.
        """
        if settings is None:
            settings = {}
        self._dict_settings = settings
        for parameter_name in self._dict_settings.keys():
            initial_lesson = GlobalTrainingStatus.get_parameter_state(
                parameter_name, StatusType.LESSON_NUM
            )
            if initial_lesson is None or not restore:
                GlobalTrainingStatus.set_parameter_state(
                    parameter_name, StatusType.LESSON_NUM, 0
                )
        self._smoothed_values: Dict[str, float] = defaultdict(float)
        for key in self._dict_settings.keys():
            self._smoothed_values[key] = 0.0
        # Update the seeds of the samplers
        self._set_sampler_seeds(run_seed)

    def _set_sampler_seeds(self, seed):
        """
        Sets the seeds for the samplers (if no seed was already present). Note that
        using the provided seed.
        """
        offset = 0
        for settings in self._dict_settings.values():
            for lesson in settings.curriculum:
                if lesson.value.seed == -1:
                    lesson.value.seed = seed + offset
                    offset += 1

    def get_minimum_reward_buffer_size(self, behavior_name: str) -> int:
        """
        Calculates the minimum size of the reward buffer a behavior must use. This
        method uses the 'min_lesson_length' sampler_parameter to determine this value.
        :param behavior_name: The name of the behavior the minimum reward buffer
        size corresponds to.
        """
        result = 1
        for settings in self._dict_settings.values():
            for lesson in settings.curriculum:
                if lesson.completion_criteria is not None:
                    if lesson.completion_criteria.behavior == behavior_name:
                        result = max(
                            result, lesson.completion_criteria.min_lesson_length
                        )
        return result

    def get_current_samplers(self) -> Dict[str, ParameterRandomizationSettings]:
        """
        Creates a dictionary from environment parameter name to their corresponding
        ParameterRandomizationSettings. If curriculum is used, the
        ParameterRandomizationSettings corresponds to the sampler of the current lesson.
        """
        samplers: Dict[str, ParameterRandomizationSettings] = {}
        for param_name, settings in self._dict_settings.items():
            lesson_num = GlobalTrainingStatus.get_parameter_state(
                param_name, StatusType.LESSON_NUM
            )
            lesson = settings.curriculum[lesson_num]
            samplers[param_name] = lesson.value
        return samplers

    def get_current_lesson_number(self) -> Dict[str, int]:
        """
        Creates a dictionary from environment parameter to the current lesson number.
        If not using curriculum, this number is always 0 for that environment parameter.
        """
        result: Dict[str, int] = {}
        for parameter_name in self._dict_settings.keys():
            result[parameter_name] = GlobalTrainingStatus.get_parameter_state(
                parameter_name, StatusType.LESSON_NUM
            )
        return result

    def log_current_lesson(self, parameter_name: Optional[str] = None) -> None:
        """
        Logs the current lesson number and sampler value of the parameter with name
        parameter_name. If no parameter_name is provided, the values and lesson
        numbers of all parameters will be displayed.
        """
        if parameter_name is not None:
            settings = self._dict_settings[parameter_name]
            lesson_number = GlobalTrainingStatus.get_parameter_state(
                parameter_name, StatusType.LESSON_NUM
            )
            lesson_name = settings.curriculum[lesson_number].name
            lesson_value = settings.curriculum[lesson_number].value
            logger.info(
                f"Parameter '{parameter_name}' is in lesson '{lesson_name}' "
                f"and has value '{lesson_value}'."
            )
        else:
            for parameter_name, settings in self._dict_settings.items():
                lesson_number = GlobalTrainingStatus.get_parameter_state(
                    parameter_name, StatusType.LESSON_NUM
                )
                lesson_name = settings.curriculum[lesson_number].name
                lesson_value = settings.curriculum[lesson_number].value
                logger.info(
                    f"Parameter '{parameter_name}' is in lesson '{lesson_name}' "
                    f"and has value '{lesson_value}'."
                )

    def update_lessons(
        self,
        trainer_steps: Dict[str, int],
        trainer_max_steps: Dict[str, int],
        trainer_reward_buffer: Dict[str, List[float]],
    ) -> Tuple[bool, bool]:
        """
        Given progress metrics, calculates if at least one environment parameter is
        in a new lesson and if at least one environment parameter requires the env
        to reset.
        :param trainer_steps: A dictionary from behavior_name to the number of training
        steps this behavior's trainer has performed.
        :param trainer_max_steps: A dictionary from behavior_name to the maximum number
        of training steps this behavior's trainer has performed.
        :param trainer_reward_buffer: A dictionary from behavior_name to the list of
        the most recent episode returns for this behavior's trainer.
        :returns: A tuple of two booleans : (True if any lesson has changed, True if
        environment needs to reset)
        """
        must_reset = False
        updated = False
        for param_name, settings in self._dict_settings.items():
            lesson_num = GlobalTrainingStatus.get_parameter_state(
                param_name, StatusType.LESSON_NUM
            )
            next_lesson_num = lesson_num + 1
            lesson = settings.curriculum[lesson_num]
            if (
                lesson.completion_criteria is not None
                and len(settings.curriculum) > next_lesson_num
            ):
                behavior_to_consider = lesson.completion_criteria.behavior
                if behavior_to_consider in trainer_steps:
                    (
                        must_increment,
                        new_smoothing,
                    ) = lesson.completion_criteria.need_increment(
                        float(trainer_steps[behavior_to_consider])
                        / float(trainer_max_steps[behavior_to_consider]),
                        trainer_reward_buffer[behavior_to_consider],
                        self._smoothed_values[param_name],
                    )
                    self._smoothed_values[param_name] = new_smoothing
                    if must_increment:
                        GlobalTrainingStatus.set_parameter_state(
                            param_name, StatusType.LESSON_NUM, next_lesson_num
                        )
                        self.log_current_lesson(param_name)
                        updated = True
                        if lesson.completion_criteria.require_reset:
                            must_reset = True
        return updated, must_reset
