# # Unity ML-Agents Toolkit
import argparse

import os
import numpy as np
import json

from typing import Callable, Optional, List, NamedTuple, Dict

import mlagents.trainers
import mlagents_envs
from mlagents import tf_utils
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.trainer_util import (
    load_config,
    TrainerFactory,
    handle_existing_directories,
)
from mlagents.trainers.stats import (
    TensorboardWriter,
    CSVWriter,
    StatsReporter,
    GaugeWriter,
    ConsoleWriter,
)
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.exception import SamplerException
from mlagents_envs.base_env import BaseEnv
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.timers import (
    hierarchical_timer,
    get_timer_tree,
    add_metadata as add_timer_metadata,
)
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


def _create_parser():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("trainer_config_path")
    argparser.add_argument(
        "--env",
        default=None,
        dest="env_path",
        help="Path to the Unity executable to train",
    )
    argparser.add_argument(
        "--curriculum",
        default=None,
        dest="curriculum_config_path",
        help="YAML file for defining the lessons for curriculum training",
    )
    argparser.add_argument(
        "--lesson",
        default=0,
        type=int,
        help="The lesson to start with when performing curriculum training",
    )
    argparser.add_argument(
        "--sampler",
        default=None,
        dest="sampler_file_path",
        help="YAML file for defining the sampler for environment parameter randomization",
    )
    argparser.add_argument(
        "--keep-checkpoints",
        default=5,
        type=int,
        help="The maximum number of model checkpoints to keep. Checkpoints are saved after the"
        "number of steps specified by the save-freq option. Once the maximum number of checkpoints"
        "has been reached, the oldest checkpoint is deleted when saving a new checkpoint.",
    )
    argparser.add_argument(
        "--load",
        default=False,
        dest="load_model",
        action="store_true",
        help=argparse.SUPPRESS,  # Deprecated but still usable for now.
    )
    argparser.add_argument(
        "--resume",
        default=False,
        dest="resume",
        action="store_true",
        help="Whether to resume training from a checkpoint. Specify a --run-id to use this option. "
        "If set, the training code loads an already trained model to initialize the neural network "
        "before resuming training. This option is only valid when the models exist, and have the same "
        "behavior names as the current agents in your scene.",
    )
    argparser.add_argument(
        "--force",
        default=False,
        dest="force",
        action="store_true",
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
    )
    argparser.add_argument(
        "--initialize-from",
        metavar="RUN_ID",
        default=None,
        help="Specify a previously saved run ID from which to initialize the model from. "
        "This can be used, for instance, to fine-tune an existing model on a new environment. "
        "Note that the previously saved models must have the same behavior parameters as your "
        "current environment.",
    )
    argparser.add_argument(
        "--save-freq",
        default=50000,
        type=int,
        help="How often (in steps) to save the model during training",
    )
    argparser.add_argument(
        "--seed",
        default=-1,
        type=int,
        help="A number to use as a seed for the random number generator used by the training code",
    )
    argparser.add_argument(
        "--train",
        default=False,
        dest="train_model",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    argparser.add_argument(
        "--inference",
        default=False,
        dest="inference",
        action="store_true",
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
    )
    argparser.add_argument(
        "--num-envs",
        default=1,
        type=int,
        help="The number of concurrent Unity environment instances to collect experiences "
        "from when training",
    )
    argparser.add_argument(
        "--no-graphics",
        default=False,
        action="store_true",
        help="Whether to run the Unity executable in no-graphics mode (i.e. without initializing "
        "the graphics driver. Use this only if your agents don't use visual observations.",
    )
    argparser.add_argument(
        "--debug",
        default=False,
        action="store_true",
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
    )
    argparser.add_argument(
        "--cpu",
        default=False,
        action="store_true",
        help="Forces training using CPU only",
    )

    argparser.add_argument("--version", action="version", version="")

    eng_conf = argparser.add_argument_group(title="Engine Configuration")
    eng_conf.add_argument(
        "--width",
        default=None,
        type=int,
        help="The width of the executable window of the environment(s) in pixels "
        "(ignored for editor training).",
    )
    eng_conf.add_argument(
        "--height",
        default=None,
        type=int,
        help="The height of the executable window of the environment(s) in pixels "
        "(ignored for editor training)",
    )
    eng_conf.add_argument(
        "--quality-level",
        default=5,
        type=int,
        help="The quality level of the environment(s). Equivalent to calling "
        "QualitySettings.SetQualityLevel in Unity.",
    )
    eng_conf.add_argument(
        "--time-scale",
        default=20,
        type=float,
        help="The time scale of the Unity environment(s). Equivalent to setting "
        "Time.timeScale in Unity.",
    )
    eng_conf.add_argument(
        "--target-frame-rate",
        default=-1,
        type=int,
        help="The target frame rate of the Unity environment(s). Equivalent to setting "
        "Application.targetFrameRate in Unity.",
    )
    eng_conf.add_argument(
        "--capture-frame-rate",
        default=60,
        type=int,
        help="The capture frame rate of the Unity environment(s). Equivalent to setting "
        "Time.captureFramerate in Unity.",
    )
    return argparser


parser = _create_parser()


class RunOptions(NamedTuple):
    trainer_config: Dict
    debug: bool = parser.get_default("debug")
    seed: int = parser.get_default("seed")
    env_path: Optional[str] = parser.get_default("env_path")
    run_id: str = parser.get_default("run_id")
    initialize_from: str = parser.get_default("initialize_from")
    load_model: bool = parser.get_default("load_model")
    resume: bool = parser.get_default("resume")
    force: bool = parser.get_default("force")
    train_model: bool = parser.get_default("train_model")
    inference: bool = parser.get_default("inference")
    save_freq: int = parser.get_default("save_freq")
    keep_checkpoints: int = parser.get_default("keep_checkpoints")
    base_port: int = parser.get_default("base_port")
    num_envs: int = parser.get_default("num_envs")
    curriculum_config: Optional[Dict] = None
    lesson: int = parser.get_default("lesson")
    no_graphics: bool = parser.get_default("no_graphics")
    multi_gpu: bool = parser.get_default("multi_gpu")
    sampler_config: Optional[Dict] = None
    env_args: Optional[List[str]] = parser.get_default("env_args")
    cpu: bool = parser.get_default("cpu")
    width: int = parser.get_default("width")
    height: int = parser.get_default("height")
    quality_level: int = parser.get_default("quality_level")
    time_scale: float = parser.get_default("time_scale")
    target_frame_rate: int = parser.get_default("target_frame_rate")
    capture_frame_rate: int = parser.get_default("capture_frame_rate")

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "RunOptions":
        """
        Takes an argparse.Namespace as specified in `parse_command_line`, loads input configuration files
        from file paths, and converts to a CommandLineOptions instance.
        :param args: collection of command-line parameters passed to mlagents-learn
        :return: CommandLineOptions representing the passed in arguments, with trainer config, curriculum and sampler
          configs loaded from files.
        """
        argparse_args = vars(args)
        trainer_config_path = argparse_args["trainer_config_path"]
        curriculum_config_path = argparse_args["curriculum_config_path"]
        argparse_args["trainer_config"] = load_config(trainer_config_path)
        if curriculum_config_path is not None:
            argparse_args["curriculum_config"] = load_config(curriculum_config_path)
        if argparse_args["sampler_file_path"] is not None:
            argparse_args["sampler_config"] = load_config(
                argparse_args["sampler_file_path"]
            )
        # Keep deprecated --load working, TODO: remove
        argparse_args["resume"] = argparse_args["resume"] or argparse_args["load_model"]
        # Since argparse accepts file paths in the config options which don't exist in CommandLineOptions,
        # these keys will need to be deleted to use the **/splat operator below.
        argparse_args.pop("sampler_file_path")
        argparse_args.pop("curriculum_config_path")
        argparse_args.pop("trainer_config_path")
        return RunOptions(**vars(args))


def get_version_string() -> str:
    # pylint: disable=no-member
    return f""" Version information:
  ml-agents: {mlagents.trainers.__version__},
  ml-agents-envs: {mlagents_envs.__version__},
  Communicator API: {UnityEnvironment.API_VERSION},
  TensorFlow: {tf_utils.tf.__version__}"""


def parse_command_line(argv: Optional[List[str]] = None) -> RunOptions:
    args = parser.parse_args(argv)
    return RunOptions.from_argparse(args)


def run_training(run_seed: int, options: RunOptions) -> None:
    """
    Launches training session.
    :param options: parsed command line arguments
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    with hierarchical_timer("run_training.setup"):
        model_path = f"./models/{options.run_id}"
        maybe_init_path = (
            f"./models/{options.initialize_from}" if options.initialize_from else None
        )
        summaries_dir = "./summaries"
        port = options.base_port

        # Configure CSV, Tensorboard Writers and StatsReporter
        # We assume reward and episode length are needed in the CSV.
        csv_writer = CSVWriter(
            summaries_dir,
            required_fields=[
                "Environment/Cumulative Reward",
                "Environment/Episode Length",
            ],
        )
        handle_existing_directories(
            model_path, summaries_dir, options.resume, options.force, maybe_init_path
        )
        tb_writer = TensorboardWriter(summaries_dir, clear_past_data=not options.resume)
        gauge_write = GaugeWriter()
        console_writer = ConsoleWriter()
        StatsReporter.add_writer(tb_writer)
        StatsReporter.add_writer(csv_writer)
        StatsReporter.add_writer(gauge_write)
        StatsReporter.add_writer(console_writer)

        if options.env_path is None:
            port = UnityEnvironment.DEFAULT_EDITOR_PORT
        env_factory = create_environment_factory(
            options.env_path, options.no_graphics, run_seed, port, options.env_args
        )
        engine_config = EngineConfig(
            width=options.width,
            height=options.height,
            quality_level=options.quality_level,
            time_scale=options.time_scale,
            target_frame_rate=options.target_frame_rate,
            capture_frame_rate=options.capture_frame_rate,
        )
        env_manager = SubprocessEnvManager(env_factory, engine_config, options.num_envs)
        maybe_meta_curriculum = try_create_meta_curriculum(
            options.curriculum_config, env_manager, options.lesson
        )
        sampler_manager, resampling_interval = create_sampler_manager(
            options.sampler_config, run_seed
        )
        trainer_factory = TrainerFactory(
            options.trainer_config,
            summaries_dir,
            options.run_id,
            model_path,
            options.keep_checkpoints,
            not options.inference,
            options.resume,
            run_seed,
            maybe_init_path,
            maybe_meta_curriculum,
            options.multi_gpu,
        )
        # Create controller and begin training.
        tc = TrainerController(
            trainer_factory,
            model_path,
            summaries_dir,
            options.run_id,
            options.save_freq,
            maybe_meta_curriculum,
            not options.inference,
            run_seed,
            sampler_manager,
            resampling_interval,
        )

    # Begin training
    try:
        tc.start_learning(env_manager)
    finally:
        env_manager.close()
        write_timing_tree(summaries_dir, options.run_id)


def write_timing_tree(summaries_dir: str, run_id: str) -> None:
    timing_path = f"{summaries_dir}/{run_id}_timers.json"
    try:
        with open(timing_path, "w") as f:
            json.dump(get_timer_tree(), f, indent=4)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save to {timing_path}. Make sure the directory exists"
        )


def create_sampler_manager(sampler_config, run_seed=None):
    resample_interval = None
    if sampler_config is not None:
        if "resampling-interval" in sampler_config:
            # Filter arguments that do not exist in the environment
            resample_interval = sampler_config.pop("resampling-interval")
            if (resample_interval <= 0) or (not isinstance(resample_interval, int)):
                raise SamplerException(
                    "Specified resampling-interval is not valid. Please provide"
                    " a positive integer value for resampling-interval"
                )

        else:
            raise SamplerException(
                "Resampling interval was not specified in the sampler file."
                " Please specify it with the 'resampling-interval' key in the sampler config file."
            )

    sampler_manager = SamplerManager(sampler_config, run_seed)
    return sampler_manager, resample_interval


def try_create_meta_curriculum(
    curriculum_config: Optional[Dict], env: SubprocessEnvManager, lesson: int
) -> Optional[MetaCurriculum]:
    if curriculum_config is None:
        return None
    else:
        meta_curriculum = MetaCurriculum(curriculum_config)
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        meta_curriculum.set_all_curricula_to_lesson_num(lesson)
        return meta_curriculum


def create_environment_factory(
    env_path: Optional[str],
    no_graphics: bool,
    seed: int,
    start_port: int,
    env_args: Optional[List[str]],
) -> Callable[[int, List[SideChannel]], BaseEnv]:
    if env_path is not None:
        launch_string = UnityEnvironment.validate_environment_path(env_path)
        if launch_string is None:
            raise UnityEnvironmentException(
                f"Couldn't launch the {env_path} environment. Provided filename does not match any environments."
            )

    def create_unity_environment(
        worker_id: int, side_channels: List[SideChannel]
    ) -> UnityEnvironment:
        # Make sure that each environment gets a different seed
        env_seed = seed + worker_id
        return UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            seed=env_seed,
            no_graphics=no_graphics,
            base_port=start_port,
            args=env_args,
            side_channels=side_channels,
        )

    return create_unity_environment


def run_cli(options: RunOptions) -> None:
    try:
        print(
            """

                        ▄▄▄▓▓▓▓
                   ╓▓▓▓▓▓▓█▓▓▓▓▓
              ,▄▄▄m▀▀▀'  ,▓▓▓▀▓▓▄                           ▓▓▓  ▓▓▌
            ▄▓▓▓▀'      ▄▓▓▀  ▓▓▓      ▄▄     ▄▄ ,▄▄ ▄▄▄▄   ,▄▄ ▄▓▓▌▄ ▄▄▄    ,▄▄
          ▄▓▓▓▀        ▄▓▓▀   ▐▓▓▌     ▓▓▌   ▐▓▓ ▐▓▓▓▀▀▀▓▓▌ ▓▓▓ ▀▓▓▌▀ ^▓▓▌  ╒▓▓▌
        ▄▓▓▓▓▓▄▄▄▄▄▄▄▄▓▓▓      ▓▀      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌   ▐▓▓▄ ▓▓▌
        ▀▓▓▓▓▀▀▀▀▀▀▀▀▀▀▓▓▄     ▓▓      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌    ▐▓▓▐▓▓
          ^█▓▓▓        ▀▓▓▄   ▐▓▓▌     ▓▓▓▓▄▓▓▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▓▄    ▓▓▓▓`
            '▀▓▓▓▄      ^▓▓▓  ▓▓▓       └▀▀▀▀ ▀▀ ^▀▀    `▀▀ `▀▀   '▀▀    ▐▓▓▌
               ▀▀▀▀▓▄▄▄   ▓▓▓▓▓▓,                                      ▓▓▓▓▀
                   `▀█▓▓▓▓▓▓▓▓▓▌
                        ¬`▀▀▀█▓

        """
        )
    except Exception:
        print("\n\n\tUnity Technologies\n")
    print(get_version_string())

    if options.debug:
        log_level = logging_util.DEBUG
    else:
        log_level = logging_util.INFO
        # disable noisy warnings from tensorflow
        tf_utils.set_warnings_enabled(False)

    logging_util.set_log_level(log_level)

    logger.debug("Configuration for this run:")
    logger.debug(json.dumps(options._asdict(), indent=4))

    # Options deprecation warnings
    if options.load_model:
        logger.warning(
            "The --load option has been deprecated. Please use the --resume option instead."
        )
    if options.train_model:
        logger.warning(
            "The --train option has been deprecated. Train mode is now the default. Use "
            "--inference to run in inference mode."
        )

    run_seed = options.seed
    if options.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Add some timer metadata
    add_timer_metadata("mlagents_version", mlagents.trainers.__version__)
    add_timer_metadata("mlagents_envs_version", mlagents_envs.__version__)
    add_timer_metadata("communication_protocol_version", UnityEnvironment.API_VERSION)
    add_timer_metadata("tensorflow_version", tf_utils.tf.__version__)

    if options.seed == -1:
        run_seed = np.random.randint(0, 10000)
    run_training(run_seed, options)


def main():
    run_cli(parse_command_line())


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
