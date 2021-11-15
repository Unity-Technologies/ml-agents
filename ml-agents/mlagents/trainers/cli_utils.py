from typing import Set, Dict, Any, TextIO
import os
import yaml
from mlagents.trainers.exception import TrainerConfigError
from mlagents_envs.environment import UnityEnvironment
import argparse
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class RaiseRemovedWarning(argparse.Action):
    """
    Internal custom Action to raise warning when argument is called.
    """

    def __init__(self, nargs=0, **kwargs):
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, arg_parser, namespace, values, option_string=None):
        logger.warning(f"The command line argument {option_string} was removed.")


class DetectDefault(argparse.Action):
    """
    Internal custom Action to help detect arguments that aren't default.
    """

    non_default_args: Set[str] = set()

    def __call__(self, arg_parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        DetectDefault.non_default_args.add(self.dest)


class DetectDefaultStoreTrue(DetectDefault):
    """
    Internal class to help detect arguments that aren't default.
    Used for store_true arguments.
    """

    def __init__(self, nargs=0, **kwargs):
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, arg_parser, namespace, values, option_string=None):
        super().__call__(arg_parser, namespace, True, option_string)


class StoreConfigFile(argparse.Action):
    """
    Custom Action to store the config file location not as part of the CLI args.
    This is because we want to maintain an equivalence between the config file's
    contents and the args themselves.
    """

    trainer_config_path: str

    def __call__(self, arg_parser, namespace, values, option_string=None):
        delattr(namespace, self.dest)
        StoreConfigFile.trainer_config_path = values


def _create_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "trainer_config_path", action=StoreConfigFile, nargs="?", default=None
    )
    argparser.add_argument(
        "--env",
        default=None,
        dest="env_path",
        help="Path to the Unity executable to train",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--load",
        default=False,
        dest="load_model",
        action=DetectDefaultStoreTrue,
        help=argparse.SUPPRESS,  # Deprecated but still usable for now.
    )
    argparser.add_argument(
        "--resume",
        default=False,
        dest="resume",
        action=DetectDefaultStoreTrue,
        help="Whether to resume training from a checkpoint. Specify a --run-id to use this option. "
        "If set, the training code loads an already trained model to initialize the neural network "
        "before resuming training. This option is only valid when the models exist, and have the same "
        "behavior names as the current agents in your scene.",
    )
    argparser.add_argument(
        "--deterministic",
        default=False,
        dest="deterministic",
        action=DetectDefaultStoreTrue,
        help="Whether to select actions deterministically in policy. `dist.mean` for continuous action "
        "space, and `dist.argmax` for deterministic action space ",
    )
    argparser.add_argument(
        "--force",
        default=False,
        dest="force",
        action=DetectDefaultStoreTrue,
        help="Whether to force-overwrite this run-id's existing summary and model data. (Without "
        "this flag, attempting to train a model with a run-id that has been used before will throw "
        "an error.",
    )
    argparser.add_argument(
        "--run-id",
        default="ppo",
        help="The identifier for the training run. This identifier is used to name the "
        "subdirectories in which the trained model and summary statistics are saved as well "
        "as the saved model itself. If you use TensorBoard to view the training statistics, "
        "always set a unique run-id for each training run. (The statistics for all runs with the "
        "same id are combined as if they were produced by a the same session.)",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--initialize-from",
        metavar="RUN_ID",
        default=None,
        help="Specify a previously saved run ID from which to initialize the model from. "
        "This can be used, for instance, to fine-tune an existing model on a new environment. "
        "Note that the previously saved models must have the same behavior parameters as your "
        "current environment.",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--seed",
        default=-1,
        type=int,
        help="A number to use as a seed for the random number generator used by the training code",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--train",
        default=False,
        dest="train_model",
        action=DetectDefaultStoreTrue,
        help=argparse.SUPPRESS,
    )
    argparser.add_argument(
        "--inference",
        default=False,
        dest="inference",
        action=DetectDefaultStoreTrue,
        help="Whether to run in Python inference mode (i.e. no training). Use with --resume to load "
        "a model trained with an existing run ID.",
    )
    argparser.add_argument(
        "--base-port",
        default=UnityEnvironment.BASE_ENVIRONMENT_PORT,
        type=int,
        help="The starting port for environment communication. Each concurrent Unity environment "
        "instance will get assigned a port sequentially, starting from the base-port. Each instance "
        "will use the port (base_port + worker_id), where the worker_id is sequential IDs given to "
        "each instance from 0 to (num_envs - 1). Note that when training using the Editor rather "
        "than an executable, the base port will be ignored.",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--num-envs",
        default=1,
        type=int,
        help="The number of concurrent Unity environment instances to collect experiences "
        "from when training",
        action=DetectDefault,
    )

    argparser.add_argument(
        "--num-areas",
        default=1,
        type=int,
        help="The number of parallel training areas in each Unity environment instance.",
        action=DetectDefault,
    )

    argparser.add_argument(
        "--debug",
        default=False,
        action=DetectDefaultStoreTrue,
        help="Whether to enable debug-level logging for some parts of the code",
    )
    argparser.add_argument(
        "--env-args",
        default=None,
        nargs=argparse.REMAINDER,
        help="Arguments passed to the Unity executable. Be aware that the standalone build will also "
        "process these as Unity Command Line Arguments. You should choose different argument names if "
        "you want to create environment-specific arguments. All arguments after this flag will be "
        "passed to the executable.",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--max-lifetime-restarts",
        default=10,
        help="The max number of times a single Unity executable can crash over its lifetime before ml-agents exits. "
        "Can be set to -1 if no limit is desired.",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--restarts-rate-limit-n",
        default=1,
        help="The maximum number of times a single Unity executable can crash over a period of time (period set in "
        "restarts-rate-limit-period-s). Can be set to -1 to not use rate limiting with restarts.",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--restarts-rate-limit-period-s",
        default=60,
        help="The period of time --restarts-rate-limit-n applies to.",
        action=DetectDefault,
    )
    argparser.add_argument(
        "--torch",
        default=False,
        action=RaiseRemovedWarning,
        help="(Removed) Use the PyTorch framework.",
    )
    argparser.add_argument(
        "--tensorflow",
        default=False,
        action=RaiseRemovedWarning,
        help="(Removed) Use the TensorFlow framework.",
    )
    argparser.add_argument(
        "--results-dir",
        default="results",
        action=DetectDefault,
        help="Results base directory",
    )

    eng_conf = argparser.add_argument_group(title="Engine Configuration")
    eng_conf.add_argument(
        "--width",
        default=84,
        type=int,
        help="The width of the executable window of the environment(s) in pixels "
        "(ignored for editor training).",
        action=DetectDefault,
    )
    eng_conf.add_argument(
        "--height",
        default=84,
        type=int,
        help="The height of the executable window of the environment(s) in pixels "
        "(ignored for editor training)",
        action=DetectDefault,
    )
    eng_conf.add_argument(
        "--quality-level",
        default=5,
        type=int,
        help="The quality level of the environment(s). Equivalent to calling "
        "QualitySettings.SetQualityLevel in Unity.",
        action=DetectDefault,
    )
    eng_conf.add_argument(
        "--time-scale",
        default=20,
        type=float,
        help="The time scale of the Unity environment(s). Equivalent to setting "
        "Time.timeScale in Unity.",
        action=DetectDefault,
    )
    eng_conf.add_argument(
        "--target-frame-rate",
        default=-1,
        type=int,
        help="The target frame rate of the Unity environment(s). Equivalent to setting "
        "Application.targetFrameRate in Unity.",
        action=DetectDefault,
    )
    eng_conf.add_argument(
        "--capture-frame-rate",
        default=60,
        type=int,
        help="The capture frame rate of the Unity environment(s). Equivalent to setting "
        "Time.captureFramerate in Unity.",
        action=DetectDefault,
    )
    eng_conf.add_argument(
        "--no-graphics",
        default=False,
        action=DetectDefaultStoreTrue,
        help="Whether to run the Unity executable in no-graphics mode (i.e. without initializing "
        "the graphics driver. Use this only if your agents don't use visual observations.",
    )

    torch_conf = argparser.add_argument_group(title="Torch Configuration")
    torch_conf.add_argument(
        "--torch-device",
        default=None,
        dest="device",
        action=DetectDefault,
        help='Settings for the default torch.device used in training, for example, "cpu", "cuda", or "cuda:0"',
    )
    return argparser


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path) as data_file:
            return _load_config(data_file)
    except OSError:
        abs_path = os.path.abspath(config_path)
        raise TrainerConfigError(f"Config file could not be found at {abs_path}.")
    except UnicodeDecodeError:
        raise TrainerConfigError(
            f"There was an error decoding Config file from {config_path}. "
            f"Make sure your file is save using UTF-8"
        )


def _load_config(fp: TextIO) -> Dict[str, Any]:
    """
    Load the yaml config from the file-like object.
    """
    try:
        return yaml.safe_load(fp)
    except yaml.parser.ParserError as e:
        raise TrainerConfigError(
            "Error parsing yaml file. Please check for formatting errors. "
            "A tool such as http://www.yamllint.com/ can be helpful with this."
        ) from e


parser = _create_parser()
