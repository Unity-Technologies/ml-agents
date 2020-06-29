import attr
import pytest
import yaml

from typing import Dict, List, Optional

from mlagents.trainers.settings import (
    RunOptions,
    TrainerSettings,
    NetworkSettings,
    PPOSettings,
    SACSettings,
    RewardSignalType,
    RewardSignalSettings,
    CuriositySettings,
    ParameterRandomizationSettings,
    UniformSettings,
    GaussianSettings,
    MultiRangeUniformSettings,
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


def check_dict_is_at_least(
    testdict1: Dict, testdict2: Dict, exceptions: Optional[List[str]] = None
) -> None:
    """
    Check if everything present in the 1st dict is the same in the second dict.
    Excludes things that the second dict has but is not present in the heirarchy of the
    1st dict. Used to compare an underspecified config dict structure (e.g. as
    would be provided by a user) with a complete one (e.g. as exported by RunOptions).
    """
    for key, val in testdict1.items():
        if exceptions is not None and key in exceptions:
            continue
        assert key in testdict2
        if isinstance(val, dict):
            check_dict_is_at_least(val, testdict2[key])
        elif isinstance(val, list):
            assert isinstance(testdict2[key], list)
            for _el0, _el1 in zip(val, testdict2[key]):
                if isinstance(_el0, dict):
                    check_dict_is_at_least(_el0, _el1)
                else:
                    assert val == testdict2[key]
        else:  # If not a dict, don't recurse into it
            assert val == testdict2[key]


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


def test_memory_settings_validation():
    with pytest.raises(TrainerConfigError):
        NetworkSettings.MemorySettings(sequence_length=128, memory_size=63)

    with pytest.raises(TrainerConfigError):
        NetworkSettings.MemorySettings(sequence_length=128, memory_size=0)


def test_parameter_randomization_structure():
    """
    Tests the ParameterRandomizationSettings structure method and all validators.
    """
    parameter_randomization_dict = {
        "mass": {
            "sampler_type": "uniform",
            "sampler_parameters": {"min_value": 1.0, "max_value": 2.0},
        },
        "scale": {
            "sampler_type": "gaussian",
            "sampler_parameters": {"mean": 1.0, "st_dev": 2.0},
        },
        "length": {
            "sampler_type": "multirangeuniform",
            "sampler_parameters": {"intervals": [[1.0, 2.0], [3.0, 4.0]]},
        },
    }
    parameter_randomization_distributions = ParameterRandomizationSettings.structure(
        parameter_randomization_dict, Dict[str, ParameterRandomizationSettings]
    )
    assert isinstance(parameter_randomization_distributions["mass"], UniformSettings)
    assert isinstance(parameter_randomization_distributions["scale"], GaussianSettings)
    assert isinstance(
        parameter_randomization_distributions["length"], MultiRangeUniformSettings
    )

    # Check invalid distribution type
    invalid_distribution_dict = {
        "mass": {
            "sampler_type": "beta",
            "sampler_parameters": {"alpha": 1.0, "beta": 2.0},
        }
    }
    with pytest.raises(ValueError):
        ParameterRandomizationSettings.structure(
            invalid_distribution_dict, Dict[str, ParameterRandomizationSettings]
        )

    # Check min less than max in uniform
    invalid_distribution_dict = {
        "mass": {
            "sampler_type": "uniform",
            "sampler_parameters": {"min_value": 2.0, "max_value": 1.0},
        }
    }
    with pytest.raises(TrainerConfigError):
        ParameterRandomizationSettings.structure(
            invalid_distribution_dict, Dict[str, ParameterRandomizationSettings]
        )

    # Check min less than max in multirange
    invalid_distribution_dict = {
        "mass": {
            "sampler_type": "multirangeuniform",
            "sampler_parameters": {"intervals": [[2.0, 1.0]]},
        }
    }
    with pytest.raises(TrainerConfigError):
        ParameterRandomizationSettings.structure(
            invalid_distribution_dict, Dict[str, ParameterRandomizationSettings]
        )

    # Check multirange has valid intervals
    invalid_distribution_dict = {
        "mass": {
            "sampler_type": "multirangeuniform",
            "sampler_parameters": {"intervals": [[1.0, 2.0], [3.0]]},
        }
    }
    with pytest.raises(TrainerConfigError):
        ParameterRandomizationSettings.structure(
            invalid_distribution_dict, Dict[str, ParameterRandomizationSettings]
        )

    # Check non-Dict input
    with pytest.raises(TrainerConfigError):
        ParameterRandomizationSettings.structure(
            "notadict", Dict[str, ParameterRandomizationSettings]
        )


@pytest.mark.parametrize("use_defaults", [True, False])
def test_exportable_settings(use_defaults):
    """
    Test that structuring and unstructuring a RunOptions object results in the same
    configuration representation.
    """
    # Try to enable as many features as possible in this test YAML to hit all the
    # edge cases. Set as much as possible as non-default values to ensure no flukes.
    test_yaml = """
    behaviors:
        3DBall:
            trainer_type: sac
            hyperparameters:
                learning_rate: 0.0004
                learning_rate_schedule: constant
                batch_size: 64
                buffer_size: 200000
                buffer_init_steps: 100
                tau: 0.006
                steps_per_update: 10.0
                save_replay_buffer: true
                init_entcoef: 0.5
                reward_signal_steps_per_update: 10.0
            network_settings:
                normalize: false
                hidden_units: 256
                num_layers: 3
                vis_encode_type: nature_cnn
                memory:
                    memory_size: 1288
                    sequence_length: 12
            reward_signals:
                extrinsic:
                    gamma: 0.999
                    strength: 1.0
                curiosity:
                    gamma: 0.999
                    strength: 1.0
            keep_checkpoints: 5
            max_steps: 500000
            time_horizon: 1000
            summary_freq: 12000
            checkpoint_interval: 1
            threaded: true
    env_settings:
        env_path: test_env_path
        env_args:
            - test_env_args1
            - test_env_args2
        base_port: 12345
        num_envs: 8
        seed: 12345
    engine_settings:
        width: 12345
        height: 12345
        quality_level: 12345
        time_scale: 12345
        target_frame_rate: 12345
        capture_frame_rate: 12345
        no_graphics: true
    checkpoint_settings:
        run_id: test_run_id
        initialize_from: test_directory
        load_model: false
        resume: true
        force: true
        train_model: false
        inference: false
    debug: true
    """
    if not use_defaults:
        loaded_yaml = yaml.safe_load(test_yaml)
        run_options = RunOptions.from_dict(yaml.safe_load(test_yaml))
    else:
        run_options = RunOptions()
    dict_export = run_options.as_dict()

    if not use_defaults:  # Don't need to check if no yaml
        check_dict_is_at_least(loaded_yaml, dict_export)

    # Re-import and verify has same elements
    run_options2 = RunOptions.from_dict(dict_export)
    second_export = run_options2.as_dict()

    # Check that the two exports are the same
    assert dict_export == second_export
