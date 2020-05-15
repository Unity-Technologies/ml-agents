import attr
import pytest

from typing import Dict

from mlagents.trainers.settings import (
    RunOptions,
    TrainerSettings,
    PPOSettings,
    SACSettings,
    RewardSignalType,
    RewardSignalSettings,
    CuriositySettings,
    TrainerType,
    strict_to_cls,
)
from mlagents.trainers.exception import TrainerConfigError


def check_if_different(testobj1: object, testobj2: object) -> None:
    assert testobj1 is not testobj2
    if attr.has(testobj1.__class__) and attr.has(testobj2.__class__):
        for key, val in attr.asdict(testobj1, recurse=False).items():
            if isinstance(val, dict) or isinstance(val, list) or attr.has(val):
                # Note: this check doesn't check the contents of mutables.
                check_if_different(val, attr.asdict(testobj2, recurse=False)[key])


def test_is_new_instance():
    """
    Verify that every instance of RunOptions() and its subclasses
    is a new instance (i.e. all factory methods are used properly.)
    """
    check_if_different(RunOptions(), RunOptions())
    check_if_different(TrainerSettings(), TrainerSettings())


def test_no_configuration():
    """
    Verify that a new config will have a PPO trainer with extrinsic rewards.
    """
    blank_runoptions = RunOptions()
    assert isinstance(blank_runoptions.behaviors["test"], TrainerSettings)
    assert isinstance(blank_runoptions.behaviors["test"].hyperparameters, PPOSettings)

    assert (
        RewardSignalType.EXTRINSIC in blank_runoptions.behaviors["test"].reward_signals
    )


def test_strict_to_cls():
    """
    Test strict structuring method.
    """

    @attr.s(auto_attribs=True)
    class TestAttrsClass:
        field1: int = 0
        field2: str = "test"

    correct_dict = {"field1": 1, "field2": "test2"}
    assert strict_to_cls(correct_dict, TestAttrsClass) == TestAttrsClass(**correct_dict)

    incorrect_dict = {"field3": 1, "field2": "test2"}

    with pytest.raises(TrainerConfigError):
        strict_to_cls(incorrect_dict, TestAttrsClass)

    with pytest.raises(TrainerConfigError):
        strict_to_cls("non_dict_input", TestAttrsClass)


def test_trainersettings_structure():
    """
    Test structuring method for TrainerSettings
    """
    trainersettings_dict = {
        "trainer_type": "sac",
        "hyperparameters": {"batch_size": 1024},
        "max_steps": 1.0,
        "reward_signals": {"curiosity": {"encoding_size": 64}},
    }
    trainer_settings = TrainerSettings.structure(trainersettings_dict, TrainerSettings)
    assert isinstance(trainer_settings.hyperparameters, SACSettings)
    assert trainer_settings.trainer_type == TrainerType.SAC
    assert isinstance(trainer_settings.max_steps, int)
    assert RewardSignalType.CURIOSITY in trainer_settings.reward_signals

    # Check invalid trainer type
    with pytest.raises(ValueError):
        trainersettings_dict = {
            "trainer_type": "puppo",
            "hyperparameters": {"batch_size": 1024},
            "max_steps": 1.0,
        }
        TrainerSettings.structure(trainersettings_dict, TrainerSettings)

    # Check invalid hyperparameter
    with pytest.raises(TrainerConfigError):
        trainersettings_dict = {
            "trainer_type": "ppo",
            "hyperparameters": {"notahyperparam": 1024},
            "max_steps": 1.0,
        }
        TrainerSettings.structure(trainersettings_dict, TrainerSettings)

    # Check non-dict
    with pytest.raises(TrainerConfigError):
        TrainerSettings.structure("notadict", TrainerSettings)

    # Check hyperparameters specified but trainer type left as default.
    # This shouldn't work as you could specify non-PPO hyperparameters.
    with pytest.raises(TrainerConfigError):
        trainersettings_dict = {"hyperparameters": {"batch_size": 1024}}
        TrainerSettings.structure(trainersettings_dict, TrainerSettings)


def test_reward_signal_structure():
    """
    Tests the RewardSignalSettings structure method. This one is special b/c
    it takes in a Dict[RewardSignalType, RewardSignalSettings].
    """
    reward_signals_dict = {
        "extrinsic": {"strength": 1.0},
        "curiosity": {"strength": 1.0},
    }
    reward_signals = RewardSignalSettings.structure(
        reward_signals_dict, Dict[RewardSignalType, RewardSignalSettings]
    )
    assert isinstance(reward_signals[RewardSignalType.EXTRINSIC], RewardSignalSettings)
    assert isinstance(reward_signals[RewardSignalType.CURIOSITY], CuriositySettings)

    # Check invalid reward signal type
    reward_signals_dict = {"puppo": {"strength": 1.0}}
    with pytest.raises(ValueError):
        RewardSignalSettings.structure(
            reward_signals_dict, Dict[RewardSignalType, RewardSignalSettings]
        )

    # Check missing GAIL demo path
    reward_signals_dict = {"gail": {"strength": 1.0}}
    with pytest.raises(TypeError):
        RewardSignalSettings.structure(
            reward_signals_dict, Dict[RewardSignalType, RewardSignalSettings]
        )

    # Check non-Dict input
    with pytest.raises(TrainerConfigError):
        RewardSignalSettings.structure(
            "notadict", Dict[RewardSignalType, RewardSignalSettings]
        )
