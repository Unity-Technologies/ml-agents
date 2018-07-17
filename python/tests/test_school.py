import pytest
from unittest.mock import patch

from unitytrainers import School


@pytest.fixture
def default_reset_parameters():
    return {"param1": 1, "param2": 1, "param3": 1}


@patch('unitytrainers.Curriculum.__init__', return_value=None)
@patch('os.listdir', return_value=['TestBrain1.json', 'TestBrain2.json'])
def test_init_school_happy_path(listdir, curriculum_mock, default_reset_parameters):
    print(curriculum_mock)
    school = School('test-school/', default_reset_parameters)

    assert len(school.brains_to_curriculums) == 2

    assert 'TestBrain1' in school.brains_to_curriculums
    assert 'TestBrain2' in school.brains_to_curriculums
