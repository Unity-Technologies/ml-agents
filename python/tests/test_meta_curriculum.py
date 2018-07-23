import pytest
from unittest.mock import patch, call, Mock

from unitytrainers.meta_curriculum import MetaCurriculum


class MetaCurriculumTest(MetaCurriculum):
    """This class allows us to test MetaCurriculum objects without calling MetaCurriculum's
    __init__ function.
    """
    def __init__(self, brains_to_curriculums):
        self._brains_to_curriculums = brains_to_curriculums


@pytest.fixture
def default_reset_parameters():
    return {'param1' : 1, 'param2' : 2, 'param3' : 3}


@pytest.fixture
def more_reset_parameters():
    return {'param4' : 4, 'param5' : 5, 'param6' : 6}


@pytest.fixture
def progresses():
    return {'TestBrain1' : 0.2, 'TestBrain2' : 0.3}


@patch('unitytrainers.Curriculum.__init__', return_value=None)
@patch('os.listdir', return_value=['TestBrain1.json', 'TestBrain2.json'])
def test_init_meta_curriculum_happy_path(listdir, mock_curriculum_init, default_reset_parameters):
    meta_curriculum = MetaCurriculum('test-meta-curriculum/', default_reset_parameters)

    assert len(meta_curriculum.brains_to_curriculums) == 2

    assert 'TestBrain1' in meta_curriculum.brains_to_curriculums
    assert 'TestBrain2' in meta_curriculum.brains_to_curriculums

    calls = [call('test-meta-curriculum/TestBrain1.json', default_reset_parameters), call('test-meta-curriculum/TestBrain2.json', default_reset_parameters)]

    mock_curriculum_init.assert_has_calls(calls)


@patch('unitytrainers.Curriculum')
@patch('unitytrainers.Curriculum')
def test_set_lesson_nums(test_brain_1_curriculum, test_brain_2_curriculum):
    meta_curriculum = MetaCurriculumTest({'TestBrain1' : test_brain_1_curriculum, 'TestBrain2' : test_brain_2_curriculum})

    meta_curriculum.lesson_nums = {'TestBrain1' : 1, 'TestBrain2' : 3}

    assert test_brain_1_curriculum.lesson_num == 1
    assert test_brain_2_curriculum.lesson_num == 3



@patch('unitytrainers.Curriculum')
@patch('unitytrainers.Curriculum')
def test_increment_lessons(test_brain_1_curriculum, test_brain_2_curriculum, progresses):
    meta_curriculum = MetaCurriculumTest({'TestBrain1' : test_brain_1_curriculum, 'TestBrain2' : test_brain_2_curriculum})

    meta_curriculum.increment_lessons(progresses)

    test_brain_1_curriculum.increment_lesson.assert_called_with(0.2)
    test_brain_2_curriculum.increment_lesson.assert_called_with(0.3)


@patch('unitytrainers.Curriculum')
@patch('unitytrainers.Curriculum')
def test_set_all_curriculums_to_lesson_num(test_brain_1_curriculum, test_brain_2_curriculum):
    meta_curriculum = MetaCurriculumTest({'TestBrain1' : test_brain_1_curriculum, 'TestBrain2' : test_brain_2_curriculum})

    meta_curriculum.set_all_curriculums_to_lesson_num(2)

    assert test_brain_1_curriculum.lesson_num == 2
    assert test_brain_2_curriculum.lesson_num == 2


@patch('unitytrainers.Curriculum')
@patch('unitytrainers.Curriculum')
def test_get_config(test_brain_1_curriculum, test_brain_2_curriculum, default_reset_parameters, more_reset_parameters):
    test_brain_1_curriculum.get_config.return_value = default_reset_parameters
    test_brain_2_curriculum.get_config.return_value = default_reset_parameters
    meta_curriculum = MetaCurriculumTest({'TestBrain1' : test_brain_1_curriculum, 'TestBrain2' : test_brain_2_curriculum})

    assert meta_curriculum.get_config() == default_reset_parameters

    test_brain_2_curriculum.get_config.return_value = more_reset_parameters

    new_reset_parameters = dict(default_reset_parameters)
    new_reset_parameters.update(more_reset_parameters)

    assert meta_curriculum.get_config() == new_reset_parameters
