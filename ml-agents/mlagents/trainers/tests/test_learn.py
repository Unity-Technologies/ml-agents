import pytest
import yaml
from unittest.mock import MagicMock, patch, mock_open
from mlagents.trainers import learn
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.learn import parse_command_line
from mlagents.trainers.cli_utils import DetectDefault
from mlagents_envs.exception import UnityEnvironmentException
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
import os.path


def basic_options(extra_args=None):
    extra_args = extra_args or {}
    args = ["basic_path"]
    if extra_args:
        args += [f"{k}={v}" for k, v in extra_args.items()]
    return parse_command_line(args)


MOCK_YAML = """
    behaviors:
        {}
    """

MOCK_INITIALIZE_YAML = """
    behaviors:
        {}
    checkpoint_settings:
        initialize_from: notuselessrun
    """

MOCK_PARAMETER_YAML = """
    behaviors:
        {}
    env_settings:
        env_path: "./oldenvfile"
        num_envs: 4
        num_areas: 4
        base_port: 4001
        seed: 9870
    checkpoint_settings:
        run_id: uselessrun
        initialize_from: notuselessrun
    debug: false
    """


@patch("mlagents.trainers.learn.write_timing_tree")
@patch("mlagents.trainers.learn.write_run_options")
@patch("mlagents.trainers.learn.validate_existing_directories")
@patch("mlagents.trainers.learn.TrainerFactory")
@patch("mlagents.trainers.learn.SubprocessEnvManager")
@patch("mlagents.trainers.learn.create_environment_factory")
@patch("mlagents.trainers.settings.load_config")
def test_run_training(
    load_config,
    create_environment_factory,
    subproc_env_mock,
    trainer_factory_mock,
    handle_dir_mock,
    write_run_options_mock,
    write_timing_tree_mock,
):
    mock_env = MagicMock()
    mock_env.external_brain_names = []
    mock_env.academy_name = "TestAcademyName"
    create_environment_factory.return_value = mock_env
    load_config.return_value = yaml.safe_load(MOCK_INITIALIZE_YAML)
    mock_param_manager = MagicMock(return_value="mock_param_manager")
    mock_init = MagicMock(return_value=None)
    with patch.object(EnvironmentParameterManager, "__new__", mock_param_manager):
        with patch.object(TrainerController, "__init__", mock_init):
            with patch.object(TrainerController, "start_learning", MagicMock()):
                options = basic_options()
                learn.run_training(0, options, 1)
                mock_init.assert_called_once_with(
                    trainer_factory_mock.return_value,
                    os.path.join("results", "ppo"),
                    "ppo",
                    "mock_param_manager",
                    True,
                    0,
                )
                handle_dir_mock.assert_called_once_with(
                    os.path.join("results", "ppo"),
                    False,
                    False,
                    os.path.join("results", "notuselessrun"),
                )
                write_timing_tree_mock.assert_called_once_with(
                    os.path.join("results", "ppo", "run_logs")
                )
                write_run_options_mock.assert_called_once_with(
                    os.path.join("results", "ppo"), options
                )
    StatsReporter.writers.clear()  # make sure there aren't any writers as added by learn.py


def test_bad_env_path():
    with pytest.raises(UnityEnvironmentException):
        factory = learn.create_environment_factory(
            env_path="/foo/bar",
            no_graphics=True,
            seed=-1,
            num_areas=1,
            start_port=8000,
            env_args=None,
            log_folder="results/log_folder",
        )
        factory(worker_id=-1, side_channels=[])


@patch("builtins.open", new_callable=mock_open, read_data=MOCK_YAML)
def test_commandline_args(mock_file):
    # No args raises
    # with pytest.raises(SystemExit):
    #     parse_command_line([])
    # Test with defaults
    opt = parse_command_line(["mytrainerpath"])
    assert opt.behaviors == {}
    assert opt.env_settings.env_path is None
    assert opt.checkpoint_settings.resume is False
    assert opt.checkpoint_settings.inference is False
    assert opt.checkpoint_settings.run_id == "ppo"
    assert opt.checkpoint_settings.initialize_from is None
    assert opt.env_settings.seed == -1
    assert opt.env_settings.base_port == 5005
    assert opt.env_settings.num_envs == 1
    assert opt.env_settings.num_areas == 1
    assert opt.engine_settings.no_graphics is False
    assert opt.debug is False
    assert opt.env_settings.env_args is None

    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--inference",
        "--run-id=myawesomerun",
        "--seed=7890",
        "--train",
        "--base-port=4004",
        "--initialize-from=testdir",
        "--num-envs=2",
        "--num-areas=2",
        "--no-graphics",
        "--debug",
    ]

    opt = parse_command_line(full_args)
    assert opt.behaviors == {}
    assert opt.env_settings.env_path == "./myenvfile"
    assert opt.checkpoint_settings.run_id == "myawesomerun"
    assert opt.checkpoint_settings.initialize_from == "testdir"
    assert opt.env_settings.seed == 7890
    assert opt.env_settings.base_port == 4004
    assert opt.env_settings.num_envs == 2
    assert opt.env_settings.num_areas == 2
    assert opt.engine_settings.no_graphics is True
    assert opt.debug is True
    assert opt.checkpoint_settings.inference is True
    assert opt.checkpoint_settings.resume is False

    # ignore init if resume is set
    full_args.append("--resume")
    opt = parse_command_line(full_args)
    assert opt.checkpoint_settings.initialize_from is None  # ignore init if resume set
    assert opt.checkpoint_settings.resume is True


@patch("builtins.open", new_callable=mock_open, read_data=MOCK_PARAMETER_YAML)
def test_yaml_args(mock_file):
    # Test with opts loaded from YAML
    DetectDefault.non_default_args.clear()
    opt = parse_command_line(["mytrainerpath"])
    assert opt.behaviors == {}
    assert opt.env_settings.env_path == "./oldenvfile"
    assert opt.checkpoint_settings.run_id == "uselessrun"
    assert opt.checkpoint_settings.initialize_from == "notuselessrun"
    assert opt.env_settings.seed == 9870
    assert opt.env_settings.base_port == 4001
    assert opt.env_settings.num_envs == 4
    assert opt.env_settings.num_areas == 4
    assert opt.engine_settings.no_graphics is False
    assert opt.debug is False
    assert opt.env_settings.env_args is None
    # Test that CLI overrides YAML
    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--resume",
        "--inference",
        "--run-id=myawesomerun",
        "--seed=7890",
        "--train",
        "--base-port=4004",
        "--num-envs=2",
        "--num-areas=2",
        "--no-graphics",
        "--debug",
        "--results-dir=myresults",
    ]

    opt = parse_command_line(full_args)
    assert opt.behaviors == {}
    assert opt.env_settings.env_path == "./myenvfile"
    assert opt.checkpoint_settings.run_id == "myawesomerun"
    assert opt.env_settings.seed == 7890
    assert opt.env_settings.base_port == 4004
    assert opt.env_settings.num_envs == 2
    assert opt.env_settings.num_areas == 2
    assert opt.engine_settings.no_graphics is True
    assert opt.debug is True
    assert opt.checkpoint_settings.inference is True
    assert opt.checkpoint_settings.resume is True
    assert opt.checkpoint_settings.results_dir == "myresults"


@patch("builtins.open", new_callable=mock_open, read_data=MOCK_YAML)
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
    assert opt.env_settings.env_args == ["--foo=bar", "--blah", "baz", "100"]
