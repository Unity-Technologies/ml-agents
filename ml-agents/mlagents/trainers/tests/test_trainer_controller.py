from unittest.mock import MagicMock, Mock, patch

import yaml
import pytest

from mlagents.trainers.trainer_controller import TrainerController
from mlagents.envs.subprocess_env_manager import EnvironmentStep
from mlagents.envs.sampler_class import SamplerManager


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        default:
            trainer: ppo
            batch_size: 32
            beta: 5.0e-3
            buffer_size: 512
            epsilon: 0.2
            gamma: 0.99
            hidden_units: 128
            lambd: 0.95
            learning_rate: 3.0e-4
            max_steps: 5.0e4
            normalize: true
            num_epoch: 5
            num_layers: 2
            time_horizon: 64
            sequence_length: 64
            summary_freq: 1000
            use_recurrent: false
            memory_size: 8
            use_curiosity: false
            curiosity_strength: 0.0
            curiosity_enc_size: 1
        """
    )


@pytest.fixture
def basic_trainer_controller():
    return TrainerController(
        trainer_factory=None,
        model_path="test_model_path",
        summaries_dir="test_summaries_dir",
        run_id="test_run_id",
        save_freq=100,
        meta_curriculum=None,
        train=True,
        training_seed=99,
        fast_simulation=True,
        sampler_manager=SamplerManager({}),
        resampling_interval=None,
    )


@patch("numpy.random.seed")
@patch("tensorflow.set_random_seed")
def test_initialization_seed(numpy_random_seed, tensorflow_set_seed):
    seed = 27
    TrainerController(
        trainer_factory=None,
        model_path="",
        summaries_dir="",
        run_id="1",
        save_freq=1,
        meta_curriculum=None,
        train=True,
        training_seed=seed,
        fast_simulation=True,
        sampler_manager=SamplerManager({}),
        resampling_interval=None,
    )
    numpy_random_seed.assert_called_with(seed)
    tensorflow_set_seed.assert_called_with(seed)


def trainer_controller_with_start_learning_mocks():
    trainer_mock = MagicMock()
    trainer_mock.get_step = 0
    trainer_mock.get_max_steps = 5
    trainer_mock.parameters = {"some": "parameter"}
    trainer_mock.write_tensorboard_text = MagicMock()

    tc = basic_trainer_controller()
    tc.initialize_trainers = MagicMock()
    tc.trainers = {"testbrain": trainer_mock}
    tc.advance = MagicMock()
    tc.trainers["testbrain"].get_step = 0

    def take_step_sideeffect(env):
        tc.trainers["testbrain"].get_step += 1
        if tc.trainers["testbrain"].get_step > 10:
            raise KeyboardInterrupt
        return 1

    tc.advance.side_effect = take_step_sideeffect

    tc._export_graph = MagicMock()
    tc._save_model = MagicMock()
    return tc, trainer_mock


@patch("tensorflow.reset_default_graph")
def test_start_learning_trains_forever_if_no_train_model(tf_reset_graph):
    tc, trainer_mock = trainer_controller_with_start_learning_mocks()
    tc.train_model = False

    tf_reset_graph.return_value = None

    env_mock = MagicMock()
    env_mock.close = MagicMock()
    env_mock.reset = MagicMock()
    env_mock.external_brains = MagicMock()

    tc.start_learning(env_mock)
    tf_reset_graph.assert_called_once()
    env_mock.reset.assert_called_once()
    assert tc.advance.call_count == 11
    tc._export_graph.assert_not_called()
    tc._save_model.assert_not_called()
    env_mock.close.assert_called_once()


@patch("tensorflow.reset_default_graph")
def test_start_learning_trains_until_max_steps_then_saves(tf_reset_graph):
    tc, trainer_mock = trainer_controller_with_start_learning_mocks()
    tf_reset_graph.return_value = None

    brain_info_mock = MagicMock()
    env_mock = MagicMock()
    env_mock.close = MagicMock()
    env_mock.reset = MagicMock(return_value=brain_info_mock)
    env_mock.external_brains = MagicMock()

    tc.start_learning(env_mock)
    tf_reset_graph.assert_called_once()
    env_mock.reset.assert_called_once()
    assert tc.advance.call_count == trainer_mock.get_max_steps + 1
    env_mock.close.assert_called_once()
    tc._save_model.assert_called_once()


def trainer_controller_with_take_step_mocks():
    trainer_mock = MagicMock()
    trainer_mock.get_step = 0
    trainer_mock.get_max_steps = 5
    trainer_mock.parameters = {"some": "parameter"}
    trainer_mock.write_tensorboard_text = MagicMock()

    tc = basic_trainer_controller()
    tc.trainers = {"testbrain": trainer_mock}

    return tc, trainer_mock


def test_take_step_adds_experiences_to_trainer_and_trains():
    tc, trainer_mock = trainer_controller_with_take_step_mocks()

    action_info_dict = {"testbrain": MagicMock()}

    old_step_info = EnvironmentStep(Mock(), Mock(), action_info_dict)
    new_step_info = EnvironmentStep(Mock(), Mock(), action_info_dict)
    trainer_mock.is_ready_update = MagicMock(return_value=True)

    env_mock = MagicMock()
    env_mock.step.return_value = [new_step_info]
    env_mock.reset.return_value = [old_step_info]

    tc.advance(env_mock)
    env_mock.reset.assert_not_called()
    env_mock.step.assert_called_once()
    trainer_mock.add_experiences.assert_called_once_with(
        new_step_info.previous_all_brain_info,
        new_step_info.current_all_brain_info,
        new_step_info.brain_name_to_action_info["testbrain"].outputs,
    )
    trainer_mock.process_experiences.assert_called_once_with(
        new_step_info.previous_all_brain_info, new_step_info.current_all_brain_info
    )
    trainer_mock.update_policy.assert_called_once()
    trainer_mock.increment_step.assert_called_once()


def test_take_step_if_not_training():
    tc, trainer_mock = trainer_controller_with_take_step_mocks()
    tc.train_model = False

    action_info_dict = {"testbrain": MagicMock()}

    old_step_info = EnvironmentStep(Mock(), Mock(), action_info_dict)
    new_step_info = EnvironmentStep(Mock(), Mock(), action_info_dict)
    trainer_mock.is_ready_update = MagicMock(return_value=False)

    env_mock = MagicMock()
    env_mock.step.return_value = [new_step_info]
    env_mock.reset.return_value = [old_step_info]

    tc.advance(env_mock)
    env_mock.reset.assert_not_called()
    env_mock.step.assert_called_once()
    trainer_mock.add_experiences.assert_called_once_with(
        new_step_info.previous_all_brain_info,
        new_step_info.current_all_brain_info,
        new_step_info.brain_name_to_action_info["testbrain"].outputs,
    )
    trainer_mock.process_experiences.assert_called_once_with(
        new_step_info.previous_all_brain_info, new_step_info.current_all_brain_info
    )
    trainer_mock.clear_update_buffer.assert_called_once()
