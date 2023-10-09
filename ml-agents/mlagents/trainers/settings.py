import os.path
import warnings

import attr
import cattr
from typing import (
    Dict,
    Optional,
    List,
    Any,
    DefaultDict,
    Mapping,
    Tuple,
    Union,
    ClassVar,
)
from enum import Enum
import collections
import argparse
import abc
import numpy as np
import math
import copy

from mlagents.trainers.cli_utils import StoreConfigFile, DetectDefault, parser
from mlagents.trainers.cli_utils import load_config
from mlagents.trainers.exception import TrainerConfigError, TrainerConfigWarning

from mlagents_envs import logging_util
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents.plugins import all_trainer_settings, all_trainer_types

logger = logging_util.get_logger(__name__)


def check_and_structure(key: str, value: Any, class_type: type) -> Any:
    attr_fields_dict = attr.fields_dict(class_type)
    if key not in attr_fields_dict:
        raise TrainerConfigError(
            f"The option {key} was specified in your YAML file for {class_type.__name__}, but is invalid."
        )
    # Apply cattr structure to the values
    return cattr.structure(value, attr_fields_dict[key].type)


def check_hyperparam_schedules(val: Dict, trainer_type: str) -> Dict:
    # Check if beta and epsilon are set. If not, set to match learning rate schedule.
    if trainer_type == "ppo" or trainer_type == "poca":
        if "beta_schedule" not in val.keys() and "learning_rate_schedule" in val.keys():
            val["beta_schedule"] = val["learning_rate_schedule"]
        if (
            "epsilon_schedule" not in val.keys()
            and "learning_rate_schedule" in val.keys()
        ):
            val["epsilon_schedule"] = val["learning_rate_schedule"]
    return val


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


def deep_update_dict(d: Dict, update_d: Mapping) -> None:
    """
    Similar to dict.update(), but works for nested dicts of dicts as well.
    """
    for key, val in update_d.items():
        if key in d and isinstance(d[key], Mapping) and isinstance(val, Mapping):
            deep_update_dict(d[key], val)
        else:
            d[key] = val


class SerializationSettings:
    convert_to_onnx = True
    onnx_opset = 9


@attr.s(auto_attribs=True)
class ExportableSettings:
    def as_dict(self):
        return cattr.unstructure(self)


class EncoderType(Enum):
    FULLY_CONNECTED = "fully_connected"
    MATCH3 = "match3"
    SIMPLE = "simple"
    NATURE_CNN = "nature_cnn"
    RESNET = "resnet"


class ScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    # TODO add support for lesson based scheduling
    # LESSON = "lesson"


class ConditioningType(Enum):
    HYPER = "hyper"
    NONE = "none"


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
    goal_conditioning_type: ConditioningType = ConditioningType.HYPER
    deterministic: bool = parser.get_default("deterministic")


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
class OnPolicyHyperparamSettings(HyperparamSettings):
    num_epoch: int = 3


@attr.s(auto_attribs=True)
class OffPolicyHyperparamSettings(HyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    reward_signal_steps_per_update: float = 4


# INTRINSIC REWARD SIGNALS #############################################################
class RewardSignalType(Enum):
    EXTRINSIC: str = "extrinsic"
    GAIL: str = "gail"
    CURIOSITY: str = "curiosity"
    RND: str = "rnd"

    def to_settings(self) -> type:
        _mapping = {
            RewardSignalType.EXTRINSIC: RewardSignalSettings,
            RewardSignalType.GAIL: GAILSettings,
            RewardSignalType.CURIOSITY: CuriositySettings,
            RewardSignalType.RND: RNDSettings,
        }
        return _mapping[self]


@attr.s(auto_attribs=True)
class RewardSignalSettings:
    gamma: float = 0.99
    strength: float = 1.0
    network_settings: NetworkSettings = attr.ib(factory=NetworkSettings)

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
            # Checks to see if user specifying deprecated encoding_size for RewardSignals.
            # If network_settings is not specified, this updates the default hidden_units
            # to the value of encoding size. If specified, this ignores encoding size and
            # uses network_settings values.
            if "encoding_size" in val:
                logger.warning(
                    "'encoding_size' was deprecated for RewardSignals. Please use network_settings."
                )
                # If network settings was not specified, use the encoding size. Otherwise, use hidden_units
                if "network_settings" not in val:
                    d_final[enum_key].network_settings.hidden_units = val[
                        "encoding_size"
                    ]
        return d_final


@attr.s(auto_attribs=True)
class GAILSettings(RewardSignalSettings):
    learning_rate: float = 3e-4
    encoding_size: Optional[int] = None
    use_actions: bool = False
    use_vail: bool = False
    demo_path: str = attr.ib(kw_only=True)


@attr.s(auto_attribs=True)
class CuriositySettings(RewardSignalSettings):
    learning_rate: float = 3e-4
    encoding_size: Optional[int] = None


@attr.s(auto_attribs=True)
class RNDSettings(RewardSignalSettings):
    learning_rate: float = 1e-4
    encoding_size: Optional[int] = None


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

    def __str__(self) -> str:
        """
        Helper method to output sampler stats to console.
        """
        raise TrainerConfigError(f"__str__ not implemented for type {self.__class__}.")

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

    def __str__(self) -> str:
        """
        Helper method to output sampler stats to console.
        """
        return f"Float: value={self.value}"

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

    def __str__(self) -> str:
        """
        Helper method to output sampler stats to console.
        """
        return f"Uniform sampler: min={self.min_value}, max={self.max_value}"

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

    def __str__(self) -> str:
        """
        Helper method to output sampler stats to console.
        """
        return f"Gaussian sampler: mean={self.mean}, stddev={self.st_dev}"

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

    def __str__(self) -> str:
        """
        Helper method to output sampler stats to console.
        """
        return f"MultiRangeUniform sampler: intervals={self.intervals}"

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


@attr.s(auto_attribs=True)
class TrainerSettings(ExportableSettings):
    default_override: ClassVar[Optional["TrainerSettings"]] = None
    trainer_type: str = "ppo"
    hyperparameters: HyperparamSettings = attr.ib()
    checkpoint_interval: int = attr.ib()

    @hyperparameters.default
    def _set_default_hyperparameters(self):
        return all_trainer_settings[self.trainer_type]()

    @checkpoint_interval.default
    def _set_default_checkpoint_interval(self):
        return 500000

    network_settings: NetworkSettings = attr.ib(factory=NetworkSettings)
    reward_signals: Dict[RewardSignalType, RewardSignalSettings] = attr.ib(
        factory=lambda: {RewardSignalType.EXTRINSIC: RewardSignalSettings()}
    )
    init_path: Optional[str] = None
    keep_checkpoints: int = 5
    even_checkpoints: bool = False
    max_steps: int = 500000
    time_horizon: int = 64
    summary_freq: int = 50000
    threaded: bool = False
    self_play: Optional[SelfPlaySettings] = None
    behavioral_cloning: Optional[BehavioralCloningSettings] = None

    cattr.register_structure_hook_func(
        lambda t: t == Dict[RewardSignalType, RewardSignalSettings],
        RewardSignalSettings.structure,
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

    @checkpoint_interval.validator
    def _set_checkpoint_interval(self, attribute, value):
        if self.even_checkpoints:
            self.checkpoint_interval = int(self.max_steps / self.keep_checkpoints)

    @staticmethod
    def dict_to_trainerdict(d: Dict, t: type) -> "TrainerSettings.DefaultTrainerDict":
        return TrainerSettings.DefaultTrainerDict(
            cattr.structure(d, Dict[str, TrainerSettings])
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

        # Check if a default_settings was specified. If so, used those as the default
        # rather than an empty dict.
        if TrainerSettings.default_override is not None:
            d_copy.update(cattr.unstructure(TrainerSettings.default_override))

        deep_update_dict(d_copy, d)

        if "framework" in d_copy:
            logger.warning("Framework option was deprecated but was specified")
            d_copy.pop("framework", None)

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
                    d_copy[key] = check_hyperparam_schedules(
                        val, d_copy["trainer_type"]
                    )
                    try:
                        d_copy[key] = strict_to_cls(
                            d_copy[key], all_trainer_settings[d_copy["trainer_type"]]
                        )
                    except KeyError:
                        raise TrainerConfigError(
                            f"Settings for trainer type {d_copy['trainer_type']} were not found"
                        )
            elif key == "max_steps":
                d_copy[key] = int(float(val))
                # In some legacy configs, max steps was specified as a float
            # elif key == "even_checkpoints":
            #     if val:
            #         d_copy["checkpoint_interval"] = int(d_copy["max_steps"] / d_copy["keep_checkpoints"])
            elif key == "trainer_type":
                if val not in all_trainer_types.keys():
                    raise TrainerConfigError(f"Invalid trainer type {val} was found")
            else:
                d_copy[key] = check_and_structure(key, val, t)
        return t(**d_copy)

    class DefaultTrainerDict(collections.defaultdict):
        def __init__(self, *args):
            # Depending on how this is called, args may have the defaultdict
            # callable at the start of the list or not. In particular, unpickling
            # will pass [TrainerSettings].
            if args and args[0] == TrainerSettings:
                super().__init__(*args)
            else:
                super().__init__(TrainerSettings, *args)
            self._config_specified = True

        def set_config_specified(self, require_config_specified: bool) -> None:
            self._config_specified = require_config_specified

        def __missing__(self, key: Any) -> "TrainerSettings":
            if TrainerSettings.default_override is not None:
                self[key] = copy.deepcopy(TrainerSettings.default_override)
            elif self._config_specified:
                raise TrainerConfigError(
                    f"The behavior name {key} has not been specified in the trainer configuration. "
                    f"Please add an entry in the configuration file for {key}, or set default_settings."
                )
            else:
                logger.warning(
                    f"Behavior name {key} does not match any behaviors specified "
                    f"in the trainer configuration file. A default configuration will be used."
                )
                self[key] = TrainerSettings()
            return self[key]


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
    results_dir: str = parser.get_default("results_dir")

    @property
    def write_path(self) -> str:
        return os.path.join(self.results_dir, self.run_id)

    @property
    def maybe_init_path(self) -> Optional[str]:
        return (
            os.path.join(self.results_dir, self.initialize_from)
            if self.initialize_from is not None
            else None
        )

    @property
    def run_logs_dir(self) -> str:
        return os.path.join(self.write_path, "run_logs")

    def prioritize_resume_init(self) -> None:
        """Prioritize explicit command line resume/init over conflicting yaml options.
        if both resume/init are set at one place use resume"""
        _non_default_args = DetectDefault.non_default_args
        if "resume" in _non_default_args:
            if self.initialize_from is not None:
                logger.warning(
                    f"Both 'resume' and 'initialize_from={self.initialize_from}' are set!"
                    f" Current run will be resumed ignoring initialization."
                )
                self.initialize_from = parser.get_default("initialize_from")
        elif "initialize_from" in _non_default_args:
            if self.resume:
                logger.warning(
                    f"Both 'resume' and 'initialize_from={self.initialize_from}' are set!"
                    f" {self.run_id} is initialized_from {self.initialize_from} and resume will be ignored."
                )
                self.resume = parser.get_default("resume")
        elif self.resume and self.initialize_from is not None:
            # no cli args but both are set in yaml file
            logger.warning(
                f"Both 'resume' and 'initialize_from={self.initialize_from}' are set in yaml file!"
                f" Current run will be resumed ignoring initialization."
            )
            self.initialize_from = parser.get_default("initialize_from")


@attr.s(auto_attribs=True)
class EnvironmentSettings:
    env_path: Optional[str] = parser.get_default("env_path")
    env_args: Optional[List[str]] = parser.get_default("env_args")
    base_port: int = parser.get_default("base_port")
    num_envs: int = attr.ib(default=parser.get_default("num_envs"))
    num_areas: int = attr.ib(default=parser.get_default("num_areas"))
    timeout_wait: int = attr.ib(default=parser.get_default("timeout_wait"))
    seed: int = parser.get_default("seed")
    max_lifetime_restarts: int = parser.get_default("max_lifetime_restarts")
    restarts_rate_limit_n: int = parser.get_default("restarts_rate_limit_n")
    restarts_rate_limit_period_s: int = parser.get_default(
        "restarts_rate_limit_period_s"
    )

    @num_envs.validator
    def validate_num_envs(self, attribute, value):
        if value > 1 and self.env_path is None:
            raise ValueError("num_envs must be 1 if env_path is not set.")

    @num_areas.validator
    def validate_num_area(self, attribute, value):
        if value <= 0:
            raise ValueError("num_areas must be set to a positive number >= 1.")


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
class TorchSettings:
    device: Optional[str] = parser.get_default("device")


@attr.s(auto_attribs=True)
class RunOptions(ExportableSettings):
    default_settings: Optional[TrainerSettings] = None
    behaviors: TrainerSettings.DefaultTrainerDict = attr.ib(
        factory=TrainerSettings.DefaultTrainerDict
    )
    env_settings: EnvironmentSettings = attr.ib(factory=EnvironmentSettings)
    engine_settings: EngineSettings = attr.ib(factory=EngineSettings)
    environment_parameters: Optional[Dict[str, EnvironmentParameterSettings]] = None
    checkpoint_settings: CheckpointSettings = attr.ib(factory=CheckpointSettings)
    torch_settings: TorchSettings = attr.ib(factory=TorchSettings)

    # These are options that are relevant to the run itself, and not the engine or environment.
    # They will be left here.
    debug: bool = parser.get_default("debug")

    # Convert to settings while making sure all fields are valid
    cattr.register_structure_hook(EnvironmentSettings, strict_to_cls)
    cattr.register_structure_hook(EngineSettings, strict_to_cls)
    cattr.register_structure_hook(CheckpointSettings, strict_to_cls)
    cattr.register_structure_hook_func(
        lambda t: t == Dict[str, EnvironmentParameterSettings],
        EnvironmentParameterSettings.structure,
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
        TrainerSettings.DefaultTrainerDict, TrainerSettings.dict_to_trainerdict
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
            "torch_settings": {},
        }
        _require_all_behaviors = True
        if config_path is not None:
            configured_dict.update(load_config(config_path))
        else:
            # If we're not loading from a file, we don't require all behavior names to be specified.
            _require_all_behaviors = False

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
                elif key in attr.fields_dict(TorchSettings):
                    configured_dict["torch_settings"][key] = val
                else:  # Base options
                    configured_dict[key] = val

        final_runoptions = RunOptions.from_dict(configured_dict)
        final_runoptions.checkpoint_settings.prioritize_resume_init()
        # Need check to bypass type checking but keep structure on dict working
        if isinstance(final_runoptions.behaviors, TrainerSettings.DefaultTrainerDict):
            # configure whether or not we should require all behavior names to be found in the config YAML
            final_runoptions.behaviors.set_config_specified(_require_all_behaviors)

        _non_default_args = DetectDefault.non_default_args

        # Prioritize the deterministic mode from the cli for deterministic actions.
        if "deterministic" in _non_default_args:
            for behaviour in final_runoptions.behaviors.keys():
                final_runoptions.behaviors[
                    behaviour
                ].network_settings.deterministic = argparse_args["deterministic"]

        return final_runoptions

    @staticmethod
    def from_dict(
        options_dict: Dict[str, Any],
    ) -> "RunOptions":
        # If a default settings was specified, set the TrainerSettings class override
        if (
            "default_settings" in options_dict.keys()
            and options_dict["default_settings"] is not None
        ):
            TrainerSettings.default_override = cattr.structure(
                options_dict["default_settings"], TrainerSettings
            )
        return cattr.structure(options_dict, RunOptions)
