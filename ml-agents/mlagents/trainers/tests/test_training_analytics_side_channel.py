import yaml
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.training_analytics_side_channel import (
    TrainingAnalyticsSideChannel,
)

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


def test_sanitize_run_options():
    run_options = RunOptions.from_dict(yaml.safe_load(test_curriculum_config_yaml))
    TrainingAnalyticsSideChannel._sanitize_run_options(run_options)
