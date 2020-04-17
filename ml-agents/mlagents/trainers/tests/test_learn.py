import pytest
from unittest.mock import MagicMock, patch, mock_open
from mlagents.trainers import learn
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.learn import parse_command_line
from mlagents_envs.exception import UnityEnvironmentException
from mlagents.trainers.stats import StatsReporter


def basic_options(extra_args=None):
    extra_args = extra_args or {}
    args = ["basic_path"]
    if extra_args:
        args += [f"{k}={v}" for k, v in extra_args.items()]
    return parse_command_line(args)


@patch("mlagents.trainers.learn.handle_existing_directories")
@patch("mlagents.trainers.learn.TrainerFactory")
@patch("mlagents.trainers.learn.SamplerManager")
@patch("mlagents.trainers.learn.SubprocessEnvManager")
@patch("mlagents.trainers.learn.create_environment_factory")
@patch("mlagents.trainers.learn.load_config")
def test_run_training(
    load_config,
    create_environment_factory,
    subproc_env_mock,
    sampler_manager_mock,
    trainer_factory_mock,
    handle_dir_mock,
):
    mock_env = MagicMock()
    mock_env.external_brain_names = []
    mock_env.academy_name = "TestAcademyName"
    create_environment_factory.return_value = mock_env
    trainer_config_mock = MagicMock()
    load_config.return_value = trainer_config_mock

    mock_init = MagicMock(return_value=None)
    with patch.object(TrainerController, "__init__", mock_init):
        with patch.object(TrainerController, "start_learning", MagicMock()):
            learn.run_training(0, basic_options())
            mock_init.assert_called_once_with(
                trainer_factory_mock.return_value,
                "./models/ppo",
                "./summaries",
                "ppo",
                50000,
                None,
                True,
                0,
                sampler_manager_mock.return_value,
                None,
            )
            handle_dir_mock.assert_called_once_with(
                "./models/ppo", "./summaries", False, False, None
            )
    StatsReporter.writers.clear()  # make sure there aren't any writers as added by learn.py


def test_bad_env_path():
    with pytest.raises(UnityEnvironmentException):
        learn.create_environment_factory(
            env_path="/foo/bar",
            no_graphics=True,
            seed=None,
            start_port=8000,
            env_args=None,
        )


@patch("builtins.open", new_callable=mock_open, read_data="{}")
def test_commandline_args(mock_file):

    # No args raises
    with pytest.raises(SystemExit):
        parse_command_line([])

    # Test with defaults
    opt = parse_command_line(["mytrainerpath"])
    assert opt.trainer_config == {}
    assert opt.env_path is None
    assert opt.curriculum_config is None
    assert opt.sampler_config is None
    assert opt.keep_checkpoints == 5
    assert opt.lesson == 0
    assert opt.resume is False
    assert opt.inference is False
    assert opt.run_id == "ppo"
    assert opt.save_freq == 50000
    assert opt.seed == -1
    assert opt.base_port == 5005
    assert opt.num_envs == 1
    assert opt.no_graphics is False
    assert opt.debug is False
    assert opt.env_args is None

    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--curriculum=./mycurriculum",
        "--sampler=./mysample",
        "--keep-checkpoints=42",
        "--lesson=3",
        "--resume",
        "--inference",
        "--run-id=myawesomerun",
        "--save-freq=123456",
        "--seed=7890",
        "--train",
        "--base-port=4004",
        "--num-envs=2",
        "--no-graphics",
        "--debug",
    ]

    opt = parse_command_line(full_args)
    assert opt.trainer_config == {}
    assert opt.env_path == "./myenvfile"
    assert opt.curriculum_config == {}
    assert opt.sampler_config == {}
    assert opt.keep_checkpoints == 42
    assert opt.lesson == 3
    assert opt.run_id == "myawesomerun"
    assert opt.save_freq == 123456
    assert opt.seed == 7890
    assert opt.base_port == 4004
    assert opt.num_envs == 2
    assert opt.no_graphics is True
    assert opt.debug is True
    assert opt.inference is True
    assert opt.resume is True


@patch("builtins.open", new_callable=mock_open, read_data="{}")
def test_env_args(mock_file):
    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--env-args",  # Everything after here will be grouped in a list
        "--foo=bar",
        "--blah",
        "baz",
        "100",
    ]

    opt = parse_command_line(full_args)
    assert opt.env_args == ["--foo=bar", "--blah", "baz", "100"]
