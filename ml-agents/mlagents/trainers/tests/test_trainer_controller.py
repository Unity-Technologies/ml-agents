from unittest.mock import MagicMock, patch
import pytest

from mlagents.tf_utils import tf
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.ghost.controller import GhostController


@pytest.fixture
def basic_trainer_controller():
    trainer_factory_mock = MagicMock()
    trainer_factory_mock.ghost_controller = GhostController()
    return TrainerController(
        trainer_factory=trainer_factory_mock,
        output_path="test_model_path",
        run_id="test_run_id",
        param_manager=EnvironmentParameterManager(),
        train=True,
        training_seed=99,
    )


@patch("numpy.random.seed")
@patch.object(tf, "set_random_seed")
def test_initialization_seed(numpy_random_seed, tensorflow_set_seed):
    seed = 27
    trainer_factory_mock = MagicMock()
    trainer_factory_mock.ghost_controller = GhostController()
    TrainerController(
        trainer_factory=trainer_factory_mock,
        output_path="",
        run_id="1",
        param_manager=None,
        train=True,
        training_seed=seed,
    )
    numpy_random_seed.assert_called_with(seed)
    tensorflow_set_seed.assert_called_with(seed)


@pytest.fixture
def trainer_controller_with_start_learning_mocks(basic_trainer_controller):
    trainer_mock = MagicMock()
    trainer_mock.get_step = 0
    trainer_mock.get_max_steps = 5
    trainer_mock.should_still_train = True
    trainer_mock.parameters = {"some": "parameter"}
    trainer_mock.write_tensorboard_text = MagicMock()

    tc = basic_trainer_controller
    tc.initialize_trainers = MagicMock()
    tc.trainers = {"testbrain": trainer_mock}
    tc.advance = MagicMock()
    tc.trainers["testbrain"].get_step = 0

    def take_step_sideeffect(env):
        tc.trainers["testbrain"].get_step += 1
        if (
            not tc.trainers["testbrain"].get_step
            <= tc.trainers["testbrain"].get_max_steps
        ):
            tc.trainers["testbrain"].should_still_train = False
        if tc.trainers["testbrain"].get_step > 10:
            raise KeyboardInterrupt
        return 1

    tc.advance.side_effect = take_step_sideeffect

    tc._save_models = MagicMock()
    return tc, trainer_mock


@patch.object(tf, "reset_default_graph")
def test_start_learning_trains_forever_if_no_train_model(
    tf_reset_graph, trainer_controller_with_start_learning_mocks
):
    tc, trainer_mock = trainer_controller_with_start_learning_mocks
    tc.train_model = False

    tf_reset_graph.return_value = None

    env_mock = MagicMock()
    env_mock.close = MagicMock()
    env_mock.reset = MagicMock()
    env_mock.training_behaviors = MagicMock()

    tc.start_learning(env_mock)
    tf_reset_graph.assert_called_once()
    env_mock.reset.assert_called_once()
    assert tc.advance.call_count == 11
    tc._save_models.assert_not_called()


@patch.object(tf, "reset_default_graph")
def test_start_learning_trains_until_max_steps_then_saves(
    tf_reset_graph, trainer_controller_with_start_learning_mocks
):
    tc, trainer_mock = trainer_controller_with_start_learning_mocks
    tf_reset_graph.return_value = None

    brain_info_mock = MagicMock()
    env_mock = MagicMock()
    env_mock.close = MagicMock()
    env_mock.reset = MagicMock(return_value=brain_info_mock)
    env_mock.training_behaviors = MagicMock()

    tc.start_learning(env_mock)
    tf_reset_graph.assert_called_once()
    env_mock.reset.assert_called_once()
    assert tc.advance.call_count == trainer_mock.get_max_steps + 1
    tc._save_models.assert_called_once()


@pytest.fixture
def trainer_controller_with_take_step_mocks(basic_trainer_controller):
    trainer_mock = MagicMock()
    trainer_mock.get_step = 0
    trainer_mock.get_max_steps = 5
    trainer_mock.parameters = {"some": "parameter"}
    trainer_mock.write_tensorboard_text = MagicMock()

    tc = basic_trainer_controller
    tc.trainers = {"testbrain": trainer_mock}
    tc.managers = {"testbrain": MagicMock()}

    return tc, trainer_mock


def test_advance_adds_experiences_to_trainer_and_trains(
    trainer_controller_with_take_step_mocks
):
    tc, trainer_mock = trainer_controller_with_take_step_mocks

    brain_name = "testbrain"

    env_mock = MagicMock()

    tc.brain_name_to_identifier[brain_name].add(brain_name)

    tc.advance(env_mock)

    env_mock.reset.assert_not_called()
    env_mock.get_steps.assert_called_once()
    env_mock.process_steps.assert_called_once()
    # May have been called many times due to thread
    trainer_mock.advance.call_count > 0
