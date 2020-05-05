import attr
import cattr
from typing import Dict, Optional, List, Any, DefaultDict, Mapping
from enum import Enum
import collections
import argparse

from mlagents.trainers.cli_utils import StoreConfigFile, DetectDefault, parser
from mlagents.trainers.cli_utils import load_config
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.models import LearningRateSchedule, EncoderType


def check_and_structure(key: str, value: Any, class_type: type) -> Any:
    attr_fields_dict = attr.fields_dict(class_type)
    if key not in attr_fields_dict:
        raise TrainerConfigError(
            f"The option {key} was specified in your YAML file for {class_type.__name__}, but is invalid."
        )
    # Apply cattr structure to the values
    return cattr.structure(value, attr_fields_dict[key].type)


def strict_to_cls(d: Mapping, t: type) -> Any:
    if d is None:
        return None
    d_copy: Dict[str, Any] = {}
    d_copy.update(d)
    for key, val in d_copy.items():
        d_copy[key] = check_and_structure(key, val, t)
    return t(**d_copy)


def trainer_settings_to_cls(d: Mapping, t: type) -> Any:
    if d is None:
        return None
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
                print(d_copy[key])
                d_copy[key] = strict_to_cls(
                    d_copy[key], TrainerSettings.to_settings(d_copy["trainer_type"])
                )
        elif key == "reward_signals":
            d_copy[key] = rewardsignal_settings_to_cls(val)
        else:
            d_copy[key] = check_and_structure(key, val, t)
    return t(**d_copy)


def rewardsignal_settings_to_cls(d: Mapping) -> Any:
    if d is None:
        return None
    d_final: Dict[RewardSignalSettings.RewardSignalType, RewardSignalSettings] = {}

    for key, val in d.items():
        try:
            enum_key = RewardSignalSettings.RewardSignalType(key)
            t = RewardSignalSettings.to_settings(enum_key)
            d_final[enum_key] = strict_to_cls(val, t)
        except KeyError:
            raise TrainerConfigError(f"Unknown reward signal type {key}")
    return d_final


def defaultdict_to_dict(d: DefaultDict) -> Dict:
    return {key: cattr.unstructure(val) for key, val in d.items()}


def dict_to_defaultdict(d: Dict, t: type) -> DefaultDict:
    return collections.defaultdict(
        TrainerSettings, cattr.structure(d, Dict[str, TrainerSettings])
    )


@attr.s(auto_attribs=True)
class NetworkSettings:
    @attr.s(auto_attribs=True)
    class MemorySettings:
        sequence_length: int = 64
        memory_size: int = 128

    normalize: bool = False
    hidden_units: int = 3
    num_layers: int = 2
    vis_encode_type: EncoderType = EncoderType.SIMPLE
    memory: Optional[MemorySettings] = None


@attr.s(auto_attribs=True)
class BehavioralCloningSettings:
    demo_path: str
    steps: int = 0
    strength: float = 1.0
    samples_per_update: int = 0
    num_epoch: Optional[int] = None
    batch_size: Optional[int] = None


@attr.s(auto_attribs=True)
class HyperparamSettings:
    batch_size: int = 1024
    buffer_size: int = 10240
    learning_rate: float = 3.0e-4
    learning_rate_schedule: LearningRateSchedule = LearningRateSchedule.CONSTANT


@attr.s(auto_attribs=True)
class PPOSettings(HyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    learning_rate_schedule: LearningRateSchedule = LearningRateSchedule.LINEAR


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


@attr.s(auto_attribs=True)
class RewardSignalSettings:
    class RewardSignalType(Enum):
        EXTRINSIC: str = "extrinsic"
        GAIL: str = "gail"
        CURIOSITY: str = "curiosity"

    @staticmethod
    def to_settings(ttype: RewardSignalType) -> type:
        _mapping = {
            RewardSignalSettings.RewardSignalType.EXTRINSIC: RewardSignalSettings,
            RewardSignalSettings.RewardSignalType.GAIL: GAILSettings,
            RewardSignalSettings.RewardSignalType.CURIOSITY: CuriositySettings,
        }
        return _mapping[ttype]

    gamma: float = 0.99
    strength: float = 1.0


@attr.s(auto_attribs=True)
class GAILSettings(RewardSignalSettings):
    encoding_size: int = 64
    learning_rate: float = 3e-4
    use_actions: bool = False
    use_vail: bool = False
    demo_path: str = attr.ib(kw_only=True)


@attr.s(auto_attribs=True)
class CuriositySettings(RewardSignalSettings):
    encoding_size: int = 128
    learning_rate: float = 3e-4


@attr.s(auto_attribs=True)
class SelfPlaySettings:
    save_steps: int = 20000
    team_change: int = attr.ib()

    @team_change.default
    def _team_change_default(self):
        # Assign team_change to about 4x save_steps
        return self.save_steps * 5

    swap_steps: int = 10000
    window: int = 10
    play_against_latest_model_ratio: float = 0.5


@attr.s(auto_attribs=True)
class TrainerSettings:
    # Edit these two fields to add new trainers #
    class TrainerType(Enum):
        PPO: str = "ppo"
        SAC: str = "sac"

    @staticmethod
    def to_settings(ttype: TrainerType) -> type:
        _mapping = {
            TrainerSettings.TrainerType.PPO: PPOSettings,
            TrainerSettings.TrainerType.SAC: SACSettings,
        }
        return _mapping[ttype]

    ###############################################

    trainer_type: TrainerType = TrainerType.PPO
    hyperparameters: HyperparamSettings = attr.ib()

    @hyperparameters.default
    def _set_default_hyperparameters(self):
        return TrainerSettings.to_settings(self.trainer_type)()

    network_settings: NetworkSettings = NetworkSettings()
    reward_signals: Dict[
        RewardSignalSettings.RewardSignalType, RewardSignalSettings
    ] = {RewardSignalSettings.RewardSignalType.EXTRINSIC: RewardSignalSettings()}
    init_path: Optional[str] = None
    output_path: str = "default"
    # TODO: Remove parser default and remove from CLI
    keep_checkpoints: int = parser.get_default("keep_checkpoints")
    max_steps: float = 500000
    time_horizon: int = 64
    summary_freq: int = 50000
    threaded: bool = True
    self_play: Optional[SelfPlaySettings] = None
    behavioral_cloning: Optional[BehavioralCloningSettings] = None

    cattr.register_structure_hook(RewardSignalSettings, rewardsignal_settings_to_cls)


@attr.s(auto_attribs=True)
class CurriculumSettings:
    class MeasureType:
        PROGRESS: str = "progress"
        REWARD: str = "reward"

    measure: str = attr.ib(default=MeasureType.REWARD)
    thresholds: List[int] = attr.Factory(list)
    min_lesson_length: int = 0
    signal_smoothing: bool = True
    parameters: Dict[str, List[float]] = attr.ib(kw_only=True)


@attr.s(auto_attribs=True)
class CheckpointSettings:
    save_freq: int = parser.get_default("save_freq")
    keep_checkpoints: int = parser.get_default("keep_checkpoints")
    run_id: str = parser.get_default("run_id")
    initialize_from: str = parser.get_default("initialize_from")
    load_model: bool = parser.get_default("load_model")
    resume: bool = parser.get_default("resume")
    force: bool = parser.get_default("force")
    train_model: bool = parser.get_default("train_model")
    inference: bool = parser.get_default("inference")
    lesson: int = parser.get_default("lesson")


@attr.s(auto_attribs=True)
class EnvironmentSettings:
    env_path: Optional[str] = parser.get_default("env_path")
    env_args: Optional[List[str]] = parser.get_default("env_args")
    base_port: int = parser.get_default("base_port")
    num_envs: int = parser.get_default("num_envs")
    seed: int = parser.get_default("seed")


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
class RunOptions:
    behaviors: DefaultDict[str, TrainerSettings] = attr.ib(
        default=attr.Factory(lambda: collections.defaultdict(TrainerSettings))
    )
    env_settings: EnvironmentSettings = EnvironmentSettings()
    engine_settings: EngineSettings = EngineSettings()
    parameter_randomization: Optional[Dict] = None
    curriculum_config: Optional[Dict[str, CurriculumSettings]] = None
    checkpoint_settings: CheckpointSettings = CheckpointSettings()

    # These are options that are relevant to the run itself, and not the engine or environment.
    # They will be left here.
    debug: bool = parser.get_default("debug")
    # Strict conversion
    cattr.register_structure_hook(EnvironmentSettings, strict_to_cls)
    cattr.register_structure_hook(EngineSettings, strict_to_cls)
    cattr.register_structure_hook(CheckpointSettings, strict_to_cls)
    cattr.register_structure_hook(TrainerSettings, trainer_settings_to_cls)
    cattr.register_structure_hook(
        DefaultDict[str, TrainerSettings], dict_to_defaultdict
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
        return cattr.structure(configured_dict, RunOptions)
