import warnings

import attr
import cattr
from typing import Dict, Optional, List, Any, DefaultDict, Mapping, Tuple, Union
from enum import Enum
import collections
import argparse
import abc
import numpy as np
import math

from mlagents.trainers.cli_utils import StoreConfigFile, DetectDefault, parser
from mlagents.trainers.cli_utils import load_config
from mlagents.trainers.exception import TrainerConfigError, TrainerConfigWarning

from mlagents_envs import logging_util
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

logger = logging_util.get_logger(__name__)


def check_and_structure(key: str, value: Any, class_type: type) -> Any:
    attr_fields_dict = attr.fields_dict(class_type)
    if key not in attr_fields_dict:
        raise TrainerConfigError(
            f"The option {key} was specified in your YAML file for {class_type.__name__}, but is invalid."
        )
    # Apply cattr structure to the values
    return cattr.structure(value, attr_fields_dict[key].type)


def strict_to_cls(d: Mapping, t: type) -> Any:
    if not isinstance(d, Mapping):
        raise TrainerConfigError(f"Unsupported config {d} for {t.__name__}.")
    d_copy: Dict[str, Any] = {}
    d_copy.update(d)
    for key, val in d_copy.items():
        d_copy[key] = check_and_structure(key, val, t)
    return t(**d_copy)


def defaultdict_to_dict(d: DefaultDict) -> Dict:
    return {key: cattr.unstructure(val) for key, val in d.items()}


class SerializationSettings:
    convert_to_barracuda = True
    convert_to_onnx = True
    onnx_opset = 9


@attr.s(auto_attribs=True)
class ExportableSettings:
    def as_dict(self):
        return cattr.unstructure(self)


class EncoderType(Enum):
    SIMPLE = "simple"
    NATURE_CNN = "nature_cnn"
    RESNET = "resnet"


class ScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


@attr.s(auto_attribs=True)
class NetworkSettings:
    @attr.s
    class MemorySettings:
        sequence_length: int = attr.ib(default=64)
        memory_size: int = attr.ib(default=128)

        @memory_size.validator
        def _check_valid_memory_size(self, attribute, value):
            if value <= 0:
                raise TrainerConfigError(
                    "When using a recurrent network, memory size must be greater than 0."
                )
            elif value % 2 != 0:
                raise TrainerConfigError(
                    "When using a recurrent network, memory size must be divisible by 2."
                )

    normalize: bool = False
    hidden_units: int = 128
    num_layers: int = 2
    vis_encode_type: EncoderType = EncoderType.SIMPLE
    memory: Optional[MemorySettings] = None


@attr.s(auto_attribs=True)
class BehavioralCloningSettings:
    demo_path: str
    steps: int = 0
    strength: float = 1.0
    samples_per_update: int = 0
    # Setting either of these to None will allow the Optimizer
    # to decide these parameters, based on Trainer hyperparams
    num_epoch: Optional[int] = None
    batch_size: Optional[int] = None


@attr.s(auto_attribs=True)
class HyperparamSettings:
    batch_size: int = 1024
    buffer_size: int = 10240
    learning_rate: float = 3.0e-4
    learning_rate_schedule: ScheduleType = ScheduleType.CONSTANT


@attr.s(auto_attribs=True)
class PPOSettings(HyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR


@attr.s(auto_attribs=True)
class SACSettings(HyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    init_entcoef: float = 1.0
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


# INTRINSIC REWARD SIGNALS #############################################################
class RewardSignalType(Enum):
    EXTRINSIC: str = "extrinsic"
    GAIL: str = "gail"
    CURIOSITY: str = "curiosity"

    def to_settings(self) -> type:
        _mapping = {
            RewardSignalType.EXTRINSIC: RewardSignalSettings,
            RewardSignalType.GAIL: GAILSettings,
            RewardSignalType.CURIOSITY: CuriositySettings,
        }
        return _mapping[self]


@attr.s(auto_attribs=True)
class RewardSignalSettings:
    gamma: float = 0.99
    strength: float = 1.0

    @staticmethod
    def structure(d: Mapping, t: type) -> Any:
        """
        Helper method to structure a Dict of RewardSignalSettings class. Meant to be registered with
        cattr.register_structure_hook() and called with cattr.structure(). This is needed to handle
        the special Enum selection of RewardSignalSettings classes.
        """
        if not isinstance(d, Mapping):
            raise TrainerConfigError(f"Unsupported reward signal configuration {d}.")
        d_final: Dict[RewardSignalType, RewardSignalSettings] = {}
        for key, val in d.items():
            enum_key = RewardSignalType(key)
            t = enum_key.to_settings()
            d_final[enum_key] = strict_to_cls(val, t)
        return d_final


@attr.s(auto_attribs=True)
class GAILSettings(RewardSignalSettings):
    encoding_size: int = 64
    learning_rate: float = 3e-4
    use_actions: bool = False
    use_vail: bool = False
    demo_path: str = attr.ib(kw_only=True)


@attr.s(auto_attribs=True)
class CuriositySettings(RewardSignalSettings):
    encoding_size: int = 64
    learning_rate: float = 3e-4


# SAMPLERS #############################################################################
class ParameterRandomizationType(Enum):
    UNIFORM: str = "uniform"
    GAUSSIAN: str = "gaussian"
    MULTIRANGEUNIFORM: str = "multirangeuniform"
    CONSTANT: str = "constant"

    def to_settings(self) -> type:
        _mapping = {
            ParameterRandomizationType.UNIFORM: UniformSettings,
            ParameterRandomizationType.GAUSSIAN: GaussianSettings,
            ParameterRandomizationType.MULTIRANGEUNIFORM: MultiRangeUniformSettings,
            ParameterRandomizationType.CONSTANT: ConstantSettings
            # Constant type is handled if a float is provided instead of a config
        }
        return _mapping[self]


@attr.s(auto_attribs=True)
class ParameterRandomizationSettings(abc.ABC):
    seed: int = parser.get_default("seed")

    @staticmethod
    def structure(
        d: Union[Mapping, float], t: type
    ) -> "ParameterRandomizationSettings":
        """
        Helper method to a ParameterRandomizationSettings class. Meant to be registered with
        cattr.register_structure_hook() and called with cattr.structure(). This is needed to handle
        the special Enum selection of ParameterRandomizationSettings classes.
        """
        if isinstance(d, (float, int)):
            return ConstantSettings(value=d)
        if not isinstance(d, Mapping):
            raise TrainerConfigError(
                f"Unsupported parameter randomization configuration {d}."
            )
        if "sampler_type" not in d:
            raise TrainerConfigError(
                f"Sampler configuration does not contain sampler_type : {d}."
            )
        if "sampler_parameters" not in d:
            raise TrainerConfigError(
                f"Sampler configuration does not contain sampler_parameters : {d}."
            )
        enum_key = ParameterRandomizationType(d["sampler_type"])
        t = enum_key.to_settings()
        return strict_to_cls(d["sampler_parameters"], t)

    @staticmethod
    def unstructure(d: "ParameterRandomizationSettings") -> Mapping:
        """
        Helper method to a ParameterRandomizationSettings class. Meant to be registered with
        cattr.register_unstructure_hook() and called with cattr.unstructure().
        """
        _reversed_mapping = {
            UniformSettings: ParameterRandomizationType.UNIFORM,
            GaussianSettings: ParameterRandomizationType.GAUSSIAN,
            MultiRangeUniformSettings: ParameterRandomizationType.MULTIRANGEUNIFORM,
            ConstantSettings: ParameterRandomizationType.CONSTANT,
        }
        sampler_type: Optional[str] = None
        for t, name in _reversed_mapping.items():
            if isinstance(d, t):
                sampler_type = name.value
        sampler_parameters = attr.asdict(d)
        return {"sampler_type": sampler_type, "sampler_parameters": sampler_parameters}

    @abc.abstractmethod
    def apply(self, key: str, env_channel: EnvironmentParametersChannel) -> None:
        """
        Helper method to send sampler settings over EnvironmentParametersChannel
        Calls the appropriate sampler type set method.
        :param key: environment parameter to be sampled
        :param env_channel: The EnvironmentParametersChannel to communicate sampler settings to environment
        """
        pass


@attr.s(auto_attribs=True)
class ConstantSettings(ParameterRandomizationSettings):
    value: float = 0.0

    def apply(self, key: str, env_channel: EnvironmentParametersChannel) -> None:
        """
        Helper method to send sampler settings over EnvironmentParametersChannel
        Calls the constant sampler type set method.
        :param key: environment parameter to be sampled
        :param env_channel: The EnvironmentParametersChannel to communicate sampler settings to environment
        """
        env_channel.set_float_parameter(key, self.value)


@attr.s(auto_attribs=True)
class UniformSettings(ParameterRandomizationSettings):
    min_value: float = attr.ib()
    max_value: float = 1.0

    @min_value.default
    def _min_value_default(self):
        return 0.0

    @min_value.validator
    def _check_min_value(self, attribute, value):
        if self.min_value > self.max_value:
            raise TrainerConfigError(
                "Minimum value is greater than maximum value in uniform sampler."
            )

    def apply(self, key: str, env_channel: EnvironmentParametersChannel) -> None:
        """
        Helper method to send sampler settings over EnvironmentParametersChannel
        Calls the uniform sampler type set method.
        :param key: environment parameter to be sampled
        :param env_channel: The EnvironmentParametersChannel to communicate sampler settings to environment
        """
        env_channel.set_uniform_sampler_parameters(
            key, self.min_value, self.max_value, self.seed
        )


@attr.s(auto_attribs=True)
class GaussianSettings(ParameterRandomizationSettings):
    mean: float = 1.0
    st_dev: float = 1.0

    def apply(self, key: str, env_channel: EnvironmentParametersChannel) -> None:
        """
        Helper method to send sampler settings over EnvironmentParametersChannel
        Calls the gaussian sampler type set method.
        :param key: environment parameter to be sampled
        :param env_channel: The EnvironmentParametersChannel to communicate sampler settings to environment
        """
        env_channel.set_gaussian_sampler_parameters(
            key, self.mean, self.st_dev, self.seed
        )


@attr.s(auto_attribs=True)
class MultiRangeUniformSettings(ParameterRandomizationSettings):
    intervals: List[Tuple[float, float]] = attr.ib()

    @intervals.default
    def _intervals_default(self):
        return [[0.0, 1.0]]

    @intervals.validator
    def _check_intervals(self, attribute, value):
        for interval in self.intervals:
            if len(interval) != 2:
                raise TrainerConfigError(
                    f"The sampling interval {interval} must contain exactly two values."
                )
            min_value, max_value = interval
            if min_value > max_value:
                raise TrainerConfigError(
                    f"Minimum value is greater than maximum value in interval {interval}."
                )

    def apply(self, key: str, env_channel: EnvironmentParametersChannel) -> None:
        """
        Helper method to send sampler settings over EnvironmentParametersChannel
        Calls the multirangeuniform sampler type set method.
        :param key: environment parameter to be sampled
        :param env_channel: The EnvironmentParametersChannel to communicate sampler settings to environment
        """
        env_channel.set_multirangeuniform_sampler_parameters(
            key, self.intervals, self.seed
        )


# ENVIRONMENT PARAMETERS ###############################################################
@attr.s(auto_attribs=True)
class CompletionCriteriaSettings:
    """
    CompletionCriteriaSettings contains the information needed to figure out if the next
    lesson must start.
    """

    class MeasureType(Enum):
        PROGRESS: str = "progress"
        REWARD: str = "reward"

    behavior: str
    measure: MeasureType = attr.ib(default=MeasureType.REWARD)
    min_lesson_length: int = 0
    signal_smoothing: bool = True
    threshold: float = attr.ib(default=0.0)
    require_reset: bool = False

    @threshold.validator
    def _check_threshold_value(self, attribute, value):
        """
        Verify that the threshold has a value between 0 and 1 when the measure is
        PROGRESS
        """
        if self.measure == self.MeasureType.PROGRESS:
            if self.threshold > 1.0:
                raise TrainerConfigError(
                    "Threshold for next lesson cannot be greater than 1 when the measure is progress."
                )
            if self.threshold < 0.0:
                raise TrainerConfigError(
                    "Threshold for next lesson cannot be negative when the measure is progress."
                )

    def need_increment(
        self, progress: float, reward_buffer: List[float], smoothing: float
    ) -> Tuple[bool, float]:
        """
        Given measures, this method returns a boolean indicating if the lesson
        needs to change now, and a float corresponding to the new smoothed value.
        """
        # Is the min number of episodes reached
        if len(reward_buffer) < self.min_lesson_length:
            return False, smoothing
        if self.measure == CompletionCriteriaSettings.MeasureType.PROGRESS:
            if progress > self.threshold:
                return True, smoothing
        if self.measure == CompletionCriteriaSettings.MeasureType.REWARD:
            if len(reward_buffer) < 1:
                return False, smoothing
            measure = np.mean(reward_buffer)
            if math.isnan(measure):
                return False, smoothing
            if self.signal_smoothing:
                measure = 0.25 * smoothing + 0.75 * measure
                smoothing = measure
            if measure > self.threshold:
                return True, smoothing
        return False, smoothing


@attr.s(auto_attribs=True)
class Lesson:
    """
    Gathers the data of one lesson for one environment parameter including its name,
    the condition that must be fullfiled for the lesson to be completed and a sampler
    for the environment parameter. If the completion_criteria is None, then this is
    the last lesson in the curriculum.
    """

    value: ParameterRandomizationSettings
    name: str
    completion_criteria: Optional[CompletionCriteriaSettings] = attr.ib(default=None)


@attr.s(auto_attribs=True)
class EnvironmentParameterSettings:
    """
    EnvironmentParameterSettings is an ordered list of lessons for one environment
    parameter.
    """

    curriculum: List[Lesson]

    @staticmethod
    def _check_lesson_chain(lessons, parameter_name):
        """
        Ensures that when using curriculum, all non-terminal lessons have a valid
        CompletionCriteria, and that the terminal lesson does not contain a CompletionCriteria.
        """
        num_lessons = len(lessons)
        for index, lesson in enumerate(lessons):
            if index < num_lessons - 1 and lesson.completion_criteria is None:
                raise TrainerConfigError(
                    f"A non-terminal lesson does not have a completion_criteria for {parameter_name}."
                )
            if index == num_lessons - 1 and lesson.completion_criteria is not None:
                warnings.warn(
                    f"Your final lesson definition contains completion_criteria for {parameter_name}."
                    f"It will be ignored.",
                    TrainerConfigWarning,
                )

    @staticmethod
    def structure(d: Mapping, t: type) -> Dict[str, "EnvironmentParameterSettings"]:
        """
        Helper method to structure a Dict of EnvironmentParameterSettings class. Meant
        to be registered with cattr.register_structure_hook() and called with
        cattr.structure().
        """
        if not isinstance(d, Mapping):
            raise TrainerConfigError(
                f"Unsupported parameter environment parameter settings {d}."
            )
        d_final: Dict[str, EnvironmentParameterSettings] = {}
        for environment_parameter, environment_parameter_config in d.items():
            if (
                isinstance(environment_parameter_config, Mapping)
                and "curriculum" in environment_parameter_config
            ):
                d_final[environment_parameter] = strict_to_cls(
                    environment_parameter_config, EnvironmentParameterSettings
                )
                EnvironmentParameterSettings._check_lesson_chain(
                    d_final[environment_parameter].curriculum, environment_parameter
                )
            else:
                sampler = ParameterRandomizationSettings.structure(
                    environment_parameter_config, ParameterRandomizationSettings
                )
                d_final[environment_parameter] = EnvironmentParameterSettings(
                    curriculum=[
                        Lesson(
                            completion_criteria=None,
                            value=sampler,
                            name=environment_parameter,
                        )
                    ]
                )
        return d_final


# TRAINERS #############################################################################
@attr.s(auto_attribs=True)
class SelfPlaySettings:
    save_steps: int = 20000
    team_change: int = attr.ib()

    @team_change.default
    def _team_change_default(self):
        # Assign team_change to about 4x save_steps
        return self.save_steps * 5

    swap_steps: int = 2000
    window: int = 10
    play_against_latest_model_ratio: float = 0.5
    initial_elo: float = 1200.0


class TrainerType(Enum):
    PPO: str = "ppo"
    SAC: str = "sac"

    def to_settings(self) -> type:
        _mapping = {TrainerType.PPO: PPOSettings, TrainerType.SAC: SACSettings}
        return _mapping[self]


class FrameworkType(Enum):
    TENSORFLOW: str = "tensorflow"
    PYTORCH: str = "pytorch"


@attr.s(auto_attribs=True)
class TrainerSettings(ExportableSettings):
    trainer_type: TrainerType = TrainerType.PPO
    hyperparameters: HyperparamSettings = attr.ib()

    @hyperparameters.default
    def _set_default_hyperparameters(self):
        return self.trainer_type.to_settings()()

    network_settings: NetworkSettings = attr.ib(factory=NetworkSettings)
    reward_signals: Dict[RewardSignalType, RewardSignalSettings] = attr.ib(
        factory=lambda: {RewardSignalType.EXTRINSIC: RewardSignalSettings()}
    )
    init_path: Optional[str] = None
    keep_checkpoints: int = 5
    checkpoint_interval: int = 500000
    max_steps: int = 500000
    time_horizon: int = 64
    summary_freq: int = 50000
    threaded: bool = True
    self_play: Optional[SelfPlaySettings] = None
    behavioral_cloning: Optional[BehavioralCloningSettings] = None
    framework: FrameworkType = FrameworkType.PYTORCH

    cattr.register_structure_hook(
        Dict[RewardSignalType, RewardSignalSettings], RewardSignalSettings.structure
    )

    @network_settings.validator
    def _check_batch_size_seq_length(self, attribute, value):
        if self.network_settings.memory is not None:
            if (
                self.network_settings.memory.sequence_length
                > self.hyperparameters.batch_size
            ):
                raise TrainerConfigError(
                    "When using memory, sequence length must be less than or equal to batch size. "
                )

    @staticmethod
    def dict_to_defaultdict(d: Dict, t: type) -> DefaultDict:
        return collections.defaultdict(
            TrainerSettings, cattr.structure(d, Dict[str, TrainerSettings])
        )

    @staticmethod
    def structure(d: Mapping, t: type) -> Any:
        """
        Helper method to structure a TrainerSettings class. Meant to be registered with
        cattr.register_structure_hook() and called with cattr.structure().
        """
        if not isinstance(d, Mapping):
            raise TrainerConfigError(f"Unsupported config {d} for {t.__name__}.")
        d_copy: Dict[str, Any] = {}
        d_copy.update(d)

        for key, val in d_copy.items():
            if attr.has(type(val)):
                # Don't convert already-converted attrs classes.
                continue
            if key == "hyperparameters":
                if "trainer_type" not in d_copy:
                    raise TrainerConfigError(
                        "Hyperparameters were specified but no trainer_type was given."
                    )
                else:
                    d_copy[key] = strict_to_cls(
                        d_copy[key], TrainerType(d_copy["trainer_type"]).to_settings()
                    )
            elif key == "max_steps":
                d_copy[key] = int(float(val))
                # In some legacy configs, max steps was specified as a float
            else:
                d_copy[key] = check_and_structure(key, val, t)
        return t(**d_copy)


# COMMAND LINE #########################################################################
@attr.s(auto_attribs=True)
class CheckpointSettings:
    run_id: str = parser.get_default("run_id")
    initialize_from: Optional[str] = parser.get_default("initialize_from")
    load_model: bool = parser.get_default("load_model")
    resume: bool = parser.get_default("resume")
    force: bool = parser.get_default("force")
    train_model: bool = parser.get_default("train_model")
    inference: bool = parser.get_default("inference")


@attr.s(auto_attribs=True)
class EnvironmentSettings:
    env_path: Optional[str] = parser.get_default("env_path")
    env_args: Optional[List[str]] = parser.get_default("env_args")
    base_port: int = parser.get_default("base_port")
    num_envs: int = attr.ib(default=parser.get_default("num_envs"))
    seed: int = parser.get_default("seed")

    @num_envs.validator
    def validate_num_envs(self, attribute, value):
        if value > 1 and self.env_path is None:
            raise ValueError("num_envs must be 1 if env_path is not set.")


@attr.s(auto_attribs=True)
class EngineSettings:
    width: int = parser.get_default("width")
    height: int = parser.get_default("height")
    quality_level: int = parser.get_default("quality_level")
    time_scale: float = parser.get_default("time_scale")
    target_frame_rate: int = parser.get_default("target_frame_rate")
    capture_frame_rate: int = parser.get_default("capture_frame_rate")
    no_graphics: bool = parser.get_default("no_graphics")


@attr.s(auto_attribs=True)
class RunOptions(ExportableSettings):
    behaviors: DefaultDict[str, TrainerSettings] = attr.ib(
        factory=lambda: collections.defaultdict(TrainerSettings)
    )
    env_settings: EnvironmentSettings = attr.ib(factory=EnvironmentSettings)
    engine_settings: EngineSettings = attr.ib(factory=EngineSettings)
    environment_parameters: Optional[Dict[str, EnvironmentParameterSettings]] = None
    checkpoint_settings: CheckpointSettings = attr.ib(factory=CheckpointSettings)

    # These are options that are relevant to the run itself, and not the engine or environment.
    # They will be left here.
    debug: bool = parser.get_default("debug")
    # Strict conversion
    cattr.register_structure_hook(EnvironmentSettings, strict_to_cls)
    cattr.register_structure_hook(EngineSettings, strict_to_cls)
    cattr.register_structure_hook(CheckpointSettings, strict_to_cls)
    cattr.register_structure_hook(
        Dict[str, EnvironmentParameterSettings], EnvironmentParameterSettings.structure
    )
    cattr.register_structure_hook(Lesson, strict_to_cls)
    cattr.register_structure_hook(
        ParameterRandomizationSettings, ParameterRandomizationSettings.structure
    )
    cattr.register_unstructure_hook(
        ParameterRandomizationSettings, ParameterRandomizationSettings.unstructure
    )
    cattr.register_structure_hook(TrainerSettings, TrainerSettings.structure)
    cattr.register_structure_hook(
        DefaultDict[str, TrainerSettings], TrainerSettings.dict_to_defaultdict
    )
    cattr.register_unstructure_hook(collections.defaultdict, defaultdict_to_dict)

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "RunOptions":
        """
        Takes an argparse.Namespace as specified in `parse_command_line`, loads input configuration files
        from file paths, and converts to a RunOptions instance.
        :param args: collection of command-line parameters passed to mlagents-learn
        :return: RunOptions representing the passed in arguments, with trainer config, curriculum and sampler
          configs loaded from files.
        """
        argparse_args = vars(args)
        config_path = StoreConfigFile.trainer_config_path

        # Load YAML
        configured_dict: Dict[str, Any] = {
            "checkpoint_settings": {},
            "env_settings": {},
            "engine_settings": {},
        }
        if config_path is not None:
            configured_dict.update(load_config(config_path))

        # Use the YAML file values for all values not specified in the CLI.
        for key in configured_dict.keys():
            # Detect bad config options
            if key not in attr.fields_dict(RunOptions):
                raise TrainerConfigError(
                    "The option {} was specified in your YAML file, but is invalid.".format(
                        key
                    )
                )
        # Override with CLI args
        # Keep deprecated --load working, TODO: remove
        argparse_args["resume"] = argparse_args["resume"] or argparse_args["load_model"]
        for key, val in argparse_args.items():
            if key in DetectDefault.non_default_args:
                if key in attr.fields_dict(CheckpointSettings):
                    configured_dict["checkpoint_settings"][key] = val
                elif key in attr.fields_dict(EnvironmentSettings):
                    configured_dict["env_settings"][key] = val
                elif key in attr.fields_dict(EngineSettings):
                    configured_dict["engine_settings"][key] = val
                else:  # Base options
                    configured_dict[key] = val

        final_runoptions = RunOptions.from_dict(configured_dict)
        return final_runoptions

    @staticmethod
    def from_dict(options_dict: Dict[str, Any]) -> "RunOptions":
        return cattr.structure(options_dict, RunOptions)
