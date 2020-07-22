import pytest
import io
import os
from unittest.mock import patch

from mlagents.trainers import trainer_util
from mlagents.trainers.cli_utils import load_config, _load_config
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.exception import TrainerConfigError, UnityTrainerException
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.tests.test_simple_rl import PPO_CONFIG
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager


@pytest.fixture
def dummy_config():
    return RunOptions(behaviors={"testbrain": PPO_CONFIG})


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
    expected_config = PPO_CONFIG

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
        trainer_factory = trainer_util.TrainerFactory(
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

    trainer_factory = trainer_util.TrainerFactory(
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
    trainer_util.handle_existing_directories(output_path, False, False)
    # Test resume with fresh path - should throw an exception.
    with pytest.raises(UnityTrainerException):
        trainer_util.handle_existing_directories(output_path, True, False)

    # make a directory
    os.mkdir(output_path)
    # Test try to train w.o. force, should complain
    with pytest.raises(UnityTrainerException):
        trainer_util.handle_existing_directories(output_path, False, False)
    # Test try to train w/ resume - should work
    trainer_util.handle_existing_directories(output_path, True, False)
    # Test try to train w/ force - should work
    trainer_util.handle_existing_directories(output_path, False, True)

    # Test initialize option
    init_path = os.path.join(tmp_path, "runid2")
    with pytest.raises(UnityTrainerException):
        trainer_util.handle_existing_directories(output_path, False, True, init_path)
    os.mkdir(init_path)
    # Should pass since the directory exists now.
    trainer_util.handle_existing_directories(output_path, False, True, init_path)
