import pytest
from unittest.mock import patch, call

from unitytrainers import School


@pytest.fixture
def default_reset_parameters():
    return {"param1": 1, "param2": 1, "param3": 1}


@patch('unitytrainers.Curriculum.__init__', return_value=None)
@patch('os.listdir', return_value=['TestBrain1.json', 'TestBrain2.json'])
def test_init_school_happy_path(listdir, mock_curriculum, default_reset_parameters):
    school = School('test-school/', default_reset_parameters)

    assert len(school.brains_to_curriculums) == 2

    assert 'TestBrain1' in school.brains_to_curriculums
    assert 'TestBrain2' in school.brains_to_curriculums

    calls = [call('test-school/TestBrain1.json', default_reset_parameters), call('test-school/TestBrain2.json', default_reset_parameters)]

    mock_curriculum.assert_has_calls(calls)
