import pytest
import yaml


from mlagents.trainers.exception import TrainerConfigError, TrainerConfigWarning
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.settings import (
    RunOptions,
    UniformSettings,
    GaussianSettings,
    ConstantSettings,
    CompletionCriteriaSettings,
)


test_sampler_config_yaml = """
environment_parameters:
  param_1:
    sampler_type: uniform
    sampler_parameters:
      min_value: 0.5
      max_value: 10
"""


def test_sampler_conversion():
    run_options = RunOptions.from_dict(yaml.safe_load(test_sampler_config_yaml))
    assert run_options.environment_parameters is not None
    assert "param_1" in run_options.environment_parameters
    lessons = run_options.environment_parameters["param_1"].curriculum
    assert len(lessons) == 1
    assert lessons[0].completion_criteria is None
    assert isinstance(lessons[0].value, UniformSettings)
    assert lessons[0].value.min_value == 0.5
    assert lessons[0].value.max_value == 10


test_sampler_and_constant_config_yaml = """
environment_parameters:
  param_1:
    sampler_type: gaussian
    sampler_parameters:
      mean: 4
      st_dev: 5
  param_2: 20
"""


def test_sampler_and_constant_conversion():
    run_options = RunOptions.from_dict(
        yaml.safe_load(test_sampler_and_constant_config_yaml)
    )
    assert "param_1" in run_options.environment_parameters
    assert "param_2" in run_options.environment_parameters
    lessons_1 = run_options.environment_parameters["param_1"].curriculum
    lessons_2 = run_options.environment_parameters["param_2"].curriculum
    # gaussian
    assert isinstance(lessons_1[0].value, GaussianSettings)
    assert lessons_1[0].value.mean == 4
    assert lessons_1[0].value.st_dev == 5
    # constant
    assert isinstance(lessons_2[0].value, ConstantSettings)
    assert lessons_2[0].value.value == 20


test_curriculum_config_yaml = """
environment_parameters:
    param_1:
      curriculum:
          - name: Lesson1
            completion_criteria:
                measure: reward
                behavior: fake_behavior
                threshold: 30
                min_lesson_length: 100
                require_reset: true
            value: 1
          - name: Lesson2
            completion_criteria:
                measure: reward
                behavior: fake_behavior
                threshold: 60
                min_lesson_length: 100
                require_reset: false
            value: 2
          - name: Lesson3
            value:
                sampler_type: uniform
                sampler_parameters:
                    min_value: 1
                    max_value: 3
"""


def test_curriculum_conversion():
    run_options = RunOptions.from_dict(yaml.safe_load(test_curriculum_config_yaml))
    assert "param_1" in run_options.environment_parameters
    lessons = run_options.environment_parameters["param_1"].curriculum
    assert len(lessons) == 3
    # First lesson
    lesson = lessons[0]
    assert lesson.completion_criteria is not None
    assert (
        lesson.completion_criteria.measure
        == CompletionCriteriaSettings.MeasureType.REWARD
    )
    assert lesson.completion_criteria.behavior == "fake_behavior"
    assert lesson.completion_criteria.threshold == 30.0
    assert lesson.completion_criteria.min_lesson_length == 100
    assert lesson.completion_criteria.require_reset
    assert isinstance(lesson.value, ConstantSettings)
    assert lesson.value.value == 1
    # Second lesson
    lesson = lessons[1]
    assert lesson.completion_criteria is not None
    assert (
        lesson.completion_criteria.measure
        == CompletionCriteriaSettings.MeasureType.REWARD
    )
    assert lesson.completion_criteria.behavior == "fake_behavior"
    assert lesson.completion_criteria.threshold == 60.0
    assert lesson.completion_criteria.min_lesson_length == 100
    assert not lesson.completion_criteria.require_reset
    assert isinstance(lesson.value, ConstantSettings)
    assert lesson.value.value == 2
    # Last lesson
    lesson = lessons[2]
    assert lesson.completion_criteria is None
    assert isinstance(lesson.value, UniformSettings)
    assert lesson.value.min_value == 1
    assert lesson.value.max_value == 3


test_bad_curriculum_no_competion_criteria_config_yaml = """
environment_parameters:
    param_1:
      curriculum:
          - name: Lesson1
            completion_criteria:
                measure: reward
                behavior: fake_behavior
                threshold: 30
                min_lesson_length: 100
                require_reset: true
            value: 1
          - name: Lesson2
            value: 2
          - name: Lesson3
            value:
                sampler_type: uniform
                sampler_parameters:
                    min_value: 1
                    max_value: 3
"""


test_bad_curriculum_all_competion_criteria_config_yaml = """
environment_parameters:
    param_1:
      curriculum:
          - name: Lesson1
            completion_criteria:
                measure: reward
                behavior: fake_behavior
                threshold: 30
                min_lesson_length: 100
                require_reset: true
            value: 1
          - name: Lesson2
            completion_criteria:
                measure: reward
                behavior: fake_behavior
                threshold: 30
                min_lesson_length: 100
                require_reset: true
            value: 2
          - name: Lesson3
            completion_criteria:
                measure: reward
                behavior: fake_behavior
                threshold: 30
                min_lesson_length: 100
                require_reset: true
            value:
                sampler_type: uniform
                sampler_parameters:
                    min_value: 1
                    max_value: 3
"""


def test_curriculum_raises_no_completion_criteria_conversion():
    with pytest.raises(TrainerConfigError):
        RunOptions.from_dict(
            yaml.safe_load(test_bad_curriculum_no_competion_criteria_config_yaml)
        )


def test_curriculum_raises_all_completion_criteria_conversion():
    with pytest.warns(TrainerConfigWarning):
        run_options = RunOptions.from_dict(
            yaml.safe_load(test_bad_curriculum_all_competion_criteria_config_yaml)
        )

    param_manager = EnvironmentParameterManager(
        run_options.environment_parameters, 1337, False
    )
    assert param_manager.update_lessons(
        trainer_steps={"fake_behavior": 500},
        trainer_max_steps={"fake_behavior": 1000},
        trainer_reward_buffer={"fake_behavior": [1000] * 101},
    ) == (True, True)
    assert param_manager.update_lessons(
        trainer_steps={"fake_behavior": 500},
        trainer_max_steps={"fake_behavior": 1000},
        trainer_reward_buffer={"fake_behavior": [1000] * 101},
    ) == (True, True)
    assert param_manager.update_lessons(
        trainer_steps={"fake_behavior": 500},
        trainer_max_steps={"fake_behavior": 1000},
        trainer_reward_buffer={"fake_behavior": [1000] * 101},
    ) == (False, False)
    assert param_manager.get_current_lesson_number() == {"param_1": 2}


test_everything_config_yaml = """
environment_parameters:
    param_1:
      curriculum:
          - name: Lesson1
            completion_criteria:
                measure: reward
                behavior: fake_behavior
                threshold: 30
                min_lesson_length: 100
                require_reset: true
            value: 1
          - name: Lesson2
            completion_criteria:
                measure: progress
                behavior: fake_behavior
                threshold: 0.5
                min_lesson_length: 100
                require_reset: false
            value: 2
          - name: Lesson3
            value:
                sampler_type: uniform
                sampler_parameters:
                    min_value: 1
                    max_value: 3
    param_2:
        sampler_type: gaussian
        sampler_parameters:
            mean: 4
            st_dev: 5
    param_3: 20
"""


def test_create_manager():
    run_options = RunOptions.from_dict(yaml.safe_load(test_everything_config_yaml))
    param_manager = EnvironmentParameterManager(
        run_options.environment_parameters, 1337, False
    )
    assert param_manager.get_minimum_reward_buffer_size("fake_behavior") == 100
    assert param_manager.get_current_lesson_number() == {
        "param_1": 0,
        "param_2": 0,
        "param_3": 0,
    }
    assert param_manager.get_current_samplers() == {
        "param_1": ConstantSettings(seed=1337, value=1),
        "param_2": GaussianSettings(seed=1337 + 3, mean=4, st_dev=5),
        "param_3": ConstantSettings(seed=1337 + 3 + 1, value=20),
    }
    # Not enough episodes completed
    assert param_manager.update_lessons(
        trainer_steps={"fake_behavior": 500},
        trainer_max_steps={"fake_behavior": 1000},
        trainer_reward_buffer={"fake_behavior": [1000] * 99},
    ) == (False, False)
    # Not enough episodes reward
    assert param_manager.update_lessons(
        trainer_steps={"fake_behavior": 500},
        trainer_max_steps={"fake_behavior": 1000},
        trainer_reward_buffer={"fake_behavior": [1] * 101},
    ) == (False, False)
    assert param_manager.update_lessons(
        trainer_steps={"fake_behavior": 500},
        trainer_max_steps={"fake_behavior": 1000},
        trainer_reward_buffer={"fake_behavior": [1000] * 101},
    ) == (True, True)
    assert param_manager.get_current_lesson_number() == {
        "param_1": 1,
        "param_2": 0,
        "param_3": 0,
    }
    param_manager_2 = EnvironmentParameterManager(
        run_options.environment_parameters, 1337, restore=True
    )
    # The use of global status should make it so that the lesson numbers are maintained
    assert param_manager_2.get_current_lesson_number() == {
        "param_1": 1,
        "param_2": 0,
        "param_3": 0,
    }
    # No reset required
    assert param_manager.update_lessons(
        trainer_steps={"fake_behavior": 700},
        trainer_max_steps={"fake_behavior": 1000},
        trainer_reward_buffer={"fake_behavior": [0] * 101},
    ) == (True, False)
    assert param_manager.get_current_samplers() == {
        "param_1": UniformSettings(seed=1337 + 2, min_value=1, max_value=3),
        "param_2": GaussianSettings(seed=1337 + 3, mean=4, st_dev=5),
        "param_3": ConstantSettings(seed=1337 + 3 + 1, value=20),
    }


test_curriculum_no_behavior_yaml = """
environment_parameters:
    param_1:
      curriculum:
          - name: Lesson1
            completion_criteria:
                measure: reward
                threshold: 30
                min_lesson_length: 100
                require_reset: true
            value: 1
          - name: Lesson2
            value: 2
"""


def test_curriculum_no_behavior():
    with pytest.raises(TypeError):
        run_options = RunOptions.from_dict(
            yaml.safe_load(test_curriculum_no_behavior_yaml)
        )
        EnvironmentParameterManager(run_options.environment_parameters, 1337, False)
