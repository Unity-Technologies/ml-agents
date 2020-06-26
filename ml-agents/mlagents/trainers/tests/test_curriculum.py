import pytest

from mlagents.trainers.exception import CurriculumConfigError
from mlagents.trainers.curriculum import Curriculum
from mlagents.trainers.settings import CurriculumSettings


dummy_curriculum_config = CurriculumSettings(
    measure="reward",
    thresholds=[10, 20, 50],
    min_lesson_length=3,
    signal_smoothing=True,
    parameters={
        "param1": [0.7, 0.5, 0.3, 0.1],
        "param2": [100, 50, 20, 15],
        "param3": [0.2, 0.3, 0.7, 0.9],
    },
)

bad_curriculum_config = CurriculumSettings(
    measure="reward",
    thresholds=[10, 20, 50],
    min_lesson_length=3,
    signal_smoothing=False,
    parameters={
        "param1": [0.7, 0.5, 0.3, 0.1],
        "param2": [100, 50, 20],
        "param3": [0.2, 0.3, 0.7, 0.9],
    },
)


@pytest.fixture
def default_reset_parameters():
    return {"param1": 1, "param2": 1, "param3": 1}


def test_init_curriculum_happy_path():
    curriculum = Curriculum("TestBrain", dummy_curriculum_config)

    assert curriculum.brain_name == "TestBrain"
    assert curriculum.lesson_num == 0
    assert curriculum.measure == "reward"


def test_increment_lesson():
    curriculum = Curriculum("TestBrain", dummy_curriculum_config)
    assert curriculum.lesson_num == 0

    curriculum.lesson_num = 1
    assert curriculum.lesson_num == 1

    assert not curriculum.increment_lesson(10)
    assert curriculum.lesson_num == 1

    assert curriculum.increment_lesson(30)
    assert curriculum.lesson_num == 2

    assert not curriculum.increment_lesson(30)
    assert curriculum.lesson_num == 2

    assert curriculum.increment_lesson(10000)
    assert curriculum.lesson_num == 3


def test_get_parameters():
    curriculum = Curriculum("TestBrain", dummy_curriculum_config)
    assert curriculum.get_config() == {"param1": 0.7, "param2": 100, "param3": 0.2}

    curriculum.lesson_num = 2
    assert curriculum.get_config() == {"param1": 0.3, "param2": 20, "param3": 0.7}
    assert curriculum.get_config(0) == {"param1": 0.7, "param2": 100, "param3": 0.2}


def test_load_bad_curriculum_file_raises_error():
    with pytest.raises(CurriculumConfigError):
        Curriculum("TestBrain", bad_curriculum_config)
