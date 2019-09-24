import io
import json
import pytest
from unittest.mock import patch, mock_open

from mlagents.trainers.exception import CurriculumConfigError, CurriculumLoadingError
from mlagents.trainers.curriculum import Curriculum


dummy_curriculum_json_str = """
    {
        "measure" : "reward",
        "thresholds" : [10, 20, 50],
        "min_lesson_length" : 3,
        "signal_smoothing" : true,
        "parameters" :
        {
            "param1" : [0.7, 0.5, 0.3, 0.1],
            "param2" : [100, 50, 20, 15],
            "param3" : [0.2, 0.3, 0.7, 0.9]
        }
    }
    """


bad_curriculum_json_str = """
    {
        "measure" : "reward",
        "thresholds" : [10, 20, 50],
        "min_lesson_length" : 3,
        "signal_smoothing" : false,
        "parameters" :
        {
            "param1" : [0.7, 0.5, 0.3, 0.1],
            "param2" : [100, 50, 20],
            "param3" : [0.2, 0.3, 0.7, 0.9]
        }
    }
    """


@pytest.fixture
def location():
    return "TestBrain.json"


@pytest.fixture
def default_reset_parameters():
    return {"param1": 1, "param2": 1, "param3": 1}


@patch("builtins.open", new_callable=mock_open, read_data=dummy_curriculum_json_str)
def test_init_curriculum_happy_path(mock_file, location, default_reset_parameters):
    curriculum = Curriculum(location, default_reset_parameters)

    assert curriculum._brain_name == "TestBrain"
    assert curriculum.lesson_num == 0
    assert curriculum.measure == "reward"


@patch("builtins.open", new_callable=mock_open, read_data=bad_curriculum_json_str)
def test_init_curriculum_bad_curriculum_raises_error(
    mock_file, location, default_reset_parameters
):
    with pytest.raises(CurriculumConfigError):
        Curriculum(location, default_reset_parameters)


@patch("builtins.open", new_callable=mock_open, read_data=dummy_curriculum_json_str)
def test_increment_lesson(mock_file, location, default_reset_parameters):
    curriculum = Curriculum(location, default_reset_parameters)
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


@patch("builtins.open", new_callable=mock_open, read_data=dummy_curriculum_json_str)
def test_get_config(mock_file):
    curriculum = Curriculum("TestBrain.json", {"param1": 1, "param2": 1, "param3": 1})
    assert curriculum.get_config() == {"param1": 0.7, "param2": 100, "param3": 0.2}

    curriculum.lesson_num = 2
    assert curriculum.get_config() == {"param1": 0.3, "param2": 20, "param3": 0.7}
    assert curriculum.get_config(0) == {"param1": 0.7, "param2": 100, "param3": 0.2}


# Test json loading and error handling. These examples don't need to valid config files.


def test_curriculum_load_good():
    expected = {"x": 1}
    value = json.dumps(expected)
    fp = io.StringIO(value)
    assert expected == Curriculum._load_curriculum(fp)


def test_curriculum_load_missing_file():
    with pytest.raises(CurriculumLoadingError):
        Curriculum.load_curriculum_file("notAValidFile.json")


def test_curriculum_load_invalid_json():
    # This isn't valid json because of the trailing comma
    contents = """
{
  "x": [1, 2, 3,]
}
"""
    fp = io.StringIO(contents)
    with pytest.raises(CurriculumLoadingError):
        Curriculum._load_curriculum(fp)
