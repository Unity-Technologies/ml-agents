import pytest
import io
import os
import yaml
from unittest.mock import patch

from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.cli_utils import load_config, _load_config
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.exception import TrainerConfigError, UnityTrainerException
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.tests.dummy_config import ppo_dummy_config
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.directory_utils import (
    validate_existing_directories,
    setup_init_path,
)


@pytest.fixture
def dummy_config():
    return RunOptions(behaviors={"testbrain": ppo_dummy_config()})


@patch("mlagents_envs.base_env.BehaviorSpec")
def test_initialize_ppo_trainer(BehaviorSpecMock, dummy_config):
    brain_name = "testbrain"
    training_behaviors = {"testbrain": BehaviorSpecMock()}
    output_path = "results_dir"
    train_model = True
    load_model = False
    seed = 11
    expected_reward_buff_cap = 1

    base_config = dummy_config.behaviors
    expected_config = ppo_dummy_config()

    def mock_constructor(
        self,
        brain,
        reward_buff_cap,
        trainer_settings,
        training,
        load,
        seed,
        artifact_path,
    ):
        assert brain == brain_name
        assert trainer_settings == expected_config
        assert reward_buff_cap == expected_reward_buff_cap
        assert training == train_model
        assert load == load_model
        assert seed == seed
        assert artifact_path == os.path.join(output_path, brain_name)

    with patch.object(PPOTrainer, "__init__", mock_constructor):
        trainer_factory = TrainerFactory(
            trainer_config=base_config,
            output_path=output_path,
            train_model=train_model,
            load_model=load_model,
            seed=seed,
            param_manager=EnvironmentParameterManager(),
        )
        trainers = {}
        for brain_name in training_behaviors.keys():
            trainers[brain_name] = trainer_factory.generate(brain_name)
        assert "testbrain" in trainers
        assert isinstance(trainers["testbrain"], PPOTrainer)


def test_handles_no_config_provided():
    """
    Make sure the trainer setup handles no configs provided at all.
    """
    brain_name = "testbrain"
    no_default_config = RunOptions().behaviors
    # Pretend this was created without a YAML file
    no_default_config.set_config_specified(False)

    trainer_factory = TrainerFactory(
        trainer_config=no_default_config,
        output_path="output_path",
        train_model=True,
        load_model=False,
        seed=42,
        param_manager=EnvironmentParameterManager(),
    )
    trainer_factory.generate(brain_name)


def test_load_config_missing_file():
    with pytest.raises(TrainerConfigError):
        load_config("thisFileDefinitelyDoesNotExist.yaml")


def test_load_config_valid_yaml():
    file_contents = """
this:
  - is fine
    """
    fp = io.StringIO(file_contents)
    res = _load_config(fp)
    assert res == {"this": ["is fine"]}


def test_load_config_invalid_yaml():
    file_contents = """
you:
  - will
- not
  - parse
    """
    with pytest.raises(TrainerConfigError):
        fp = io.StringIO(file_contents)
        _load_config(fp)


def test_existing_directories(tmp_path):
    output_path = os.path.join(tmp_path, "runid")
    # Test fresh new unused path - should do nothing.
    validate_existing_directories(output_path, False, False)
    # Test resume with fresh path - should throw an exception.
    with pytest.raises(UnityTrainerException):
        validate_existing_directories(output_path, True, False)

    # make a directory
    os.mkdir(output_path)
    # Test try to train w.o. force, should complain
    with pytest.raises(UnityTrainerException):
        validate_existing_directories(output_path, False, False)
    # Test try to train w/ resume - should work
    validate_existing_directories(output_path, True, False)
    # Test try to train w/ force - should work
    validate_existing_directories(output_path, False, True)

    # Test initialize option
    init_path = os.path.join(tmp_path, "runid2")
    with pytest.raises(UnityTrainerException):
        validate_existing_directories(output_path, False, True, init_path)
    os.mkdir(init_path)
    # Should pass since the directory exists now.
    validate_existing_directories(output_path, False, True, init_path)


@pytest.mark.parametrize("dir_exists", [True, False])
def test_setup_init_path(tmpdir, dir_exists):
    """

    :return:
    """
    test_yaml = """
    behaviors:
        BigWallJump:
            init_path: BigWallJump-6540981.pt #full path
            trainer_type: ppo
        MediumWallJump:
            init_path: {}/test_setup_init_path_results/test_run_id/MediumWallJump/checkpoint.pt
            trainer_type: ppo
        SmallWallJump:
            trainer_type: ppo
    checkpoint_settings:
        run_id: test_run_id
        initialize_from: test_run_id
    """.format(
        tmpdir
    )
    run_options = RunOptions.from_dict(yaml.safe_load(test_yaml))
    if dir_exists:
        init_path = tmpdir.mkdir("test_setup_init_path_results").mkdir("test_run_id")
        big = init_path.mkdir("BigWallJump").join("BigWallJump-6540981.pt")
        big.write("content")
        med = init_path.mkdir("MediumWallJump").join("checkpoint.pt")
        med.write("content")
        small = init_path.mkdir("SmallWallJump").join("checkpoint.pt")
        small.write("content")

        setup_init_path(run_options.behaviors, init_path)
        assert run_options.behaviors["BigWallJump"].init_path == big
        assert run_options.behaviors["MediumWallJump"].init_path == med
        assert run_options.behaviors["SmallWallJump"].init_path == small
    else:
        # don't make dirs and fail
        with pytest.raises(UnityTrainerException):
            setup_init_path(
                run_options.behaviors, run_options.checkpoint_settings.maybe_init_path
            )
