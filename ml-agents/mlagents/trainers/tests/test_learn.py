import unittest.mock as mock
import pytest
from unittest.mock import *
from mlagents.trainers import learn, TrainerController


@pytest.fixture
def basic_options():
    return {
        "--docker-target-name": "None",
        "--env": "None",
        "--run-id": "ppo",
        "--load": False,
        "--train": False,
        "--save-freq": "50000",
        "--keep-checkpoints": "5",
        "--base-port": "5005",
        "--num-envs": "1",
        "--curriculum": "None",
        "--lesson": "0",
        "--slow": False,
        "--no-graphics": False,
        "<trainer-config-path>": "basic_path",
        "--debug": False,
    }


@patch("mlagents.trainers.learn.SubprocessUnityEnvironment")
@patch("mlagents.trainers.learn.create_environment_factory")
@patch("mlagents.trainers.learn.load_config")
def test_run_training(load_config, create_environment_factory, subproc_env_mock):
    mock_env = MagicMock()
    mock_env.external_brain_names = []
    mock_env.academy_name = "TestAcademyName"
    create_environment_factory.return_value = mock_env
    trainer_config_mock = MagicMock()
    load_config.return_value = trainer_config_mock

    mock_init = MagicMock(return_value=None)
    with patch.object(TrainerController, "__init__", mock_init):
        with patch.object(TrainerController, "start_learning", MagicMock()):
            learn.run_training(0, 0, basic_options(), MagicMock())
            mock_init.assert_called_once_with(
                "./models/ppo-0",
                "./summaries",
                "ppo-0",
                50000,
                None,
                False,
                False,
                5,
                0,
                subproc_env_mock.return_value.external_brains,
                0,
                True,
            )


@patch("mlagents.trainers.learn.SubprocessUnityEnvironment")
@patch("mlagents.trainers.learn.create_environment_factory")
@patch("mlagents.trainers.learn.load_config")
def test_docker_target_path(load_config, create_environment_factory, subproc_env_mock):
    mock_env = MagicMock()
    mock_env.external_brain_names = []
    mock_env.academy_name = "TestAcademyName"
    create_environment_factory.return_value = mock_env
    trainer_config_mock = MagicMock()
    load_config.return_value = trainer_config_mock

    options_with_docker_target = basic_options()
    options_with_docker_target["--docker-target-name"] = "dockertarget"

    mock_init = MagicMock(return_value=None)
    with patch.object(TrainerController, "__init__", mock_init):
        with patch.object(TrainerController, "start_learning", MagicMock()):
            learn.run_training(0, 0, options_with_docker_target, MagicMock())
            mock_init.assert_called_once()
            assert mock_init.call_args[0][0] == "/dockertarget/models/ppo-0"
            assert mock_init.call_args[0][1] == "/dockertarget/summaries"
