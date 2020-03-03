import pytest
from unittest.mock import patch, Mock

from mlagents.trainers.meta_curriculum import MetaCurriculum
import json
import yaml

from mlagents.trainers.tests.simple_test_envs import Simple1DEnvironment
from mlagents.trainers.tests.test_simple_rl import _check_environment_trains, BRAIN_NAME
from mlagents.trainers.tests.test_curriculum import dummy_curriculum_json_str


@pytest.fixture
def measure_vals():
    return {"Brain1": 0.2, "Brain2": 0.3}


@pytest.fixture
def reward_buff_sizes():
    return {"Brain1": 7, "Brain2": 8}


def test_curriculum_config(param_name="test_param1", min_lesson_length=100):
    return {
        "measure": "progress",
        "thresholds": [0.1, 0.3, 0.5],
        "min_lesson_length": min_lesson_length,
        "signal_smoothing": True,
        "parameters": {f"{param_name}": [0.0, 4.0, 6.0, 8.0]},
    }


test_meta_curriculum_config = {
    "Brain1": test_curriculum_config("test_param1"),
    "Brain2": test_curriculum_config("test_param2"),
}


def test_set_lesson_nums():
    meta_curriculum = MetaCurriculum(test_meta_curriculum_config)
    meta_curriculum.lesson_nums = {"Brain1": 1, "Brain2": 3}

    assert meta_curriculum.brains_to_curricula["Brain1"].lesson_num == 1
    assert meta_curriculum.brains_to_curricula["Brain2"].lesson_num == 3


def test_increment_lessons(measure_vals):
    meta_curriculum = MetaCurriculum(test_meta_curriculum_config)
    meta_curriculum.brains_to_curricula["Brain1"] = Mock()
    meta_curriculum.brains_to_curricula["Brain2"] = Mock()

    meta_curriculum.increment_lessons(measure_vals)

    meta_curriculum.brains_to_curricula["Brain1"].increment_lesson.assert_called_with(
        0.2
    )
    meta_curriculum.brains_to_curricula["Brain2"].increment_lesson.assert_called_with(
        0.3
    )


@patch("mlagents.trainers.curriculum.Curriculum")
@patch("mlagents.trainers.curriculum.Curriculum")
def test_increment_lessons_with_reward_buff_sizes(
    curriculum_a, curriculum_b, measure_vals, reward_buff_sizes
):
    curriculum_a.min_lesson_length = 5
    curriculum_b.min_lesson_length = 10
    meta_curriculum = MetaCurriculum(test_meta_curriculum_config)
    meta_curriculum.brains_to_curricula["Brain1"] = curriculum_a
    meta_curriculum.brains_to_curricula["Brain2"] = curriculum_b

    meta_curriculum.increment_lessons(measure_vals, reward_buff_sizes=reward_buff_sizes)

    curriculum_a.increment_lesson.assert_called_with(0.2)
    curriculum_b.increment_lesson.assert_not_called()


def test_set_all_curriculums_to_lesson_num():
    meta_curriculum = MetaCurriculum(test_meta_curriculum_config)

    meta_curriculum.set_all_curricula_to_lesson_num(2)

    assert meta_curriculum.brains_to_curricula["Brain1"].lesson_num == 2
    assert meta_curriculum.brains_to_curricula["Brain2"].lesson_num == 2


def test_get_config():
    meta_curriculum = MetaCurriculum(test_meta_curriculum_config)
    assert meta_curriculum.get_config() == {"test_param1": 0.0, "test_param2": 0.0}


TRAINER_CONFIG = """
    default:
        trainer: ppo
        batch_size: 16
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 5.0e-3
        max_steps: 100
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 50
        use_recurrent: false
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
    """


@pytest.mark.parametrize("curriculum_brain_name", [BRAIN_NAME, "WrongBrainName"])
def test_simple_metacurriculum(curriculum_brain_name):
    env = Simple1DEnvironment([BRAIN_NAME], use_discrete=False)
    curriculum_config = json.loads(dummy_curriculum_json_str)
    mc = MetaCurriculum({curriculum_brain_name: curriculum_config})
    trainer_config = yaml.safe_load(TRAINER_CONFIG)
    _check_environment_trains(
        env, trainer_config, meta_curriculum=mc, success_threshold=None
    )
