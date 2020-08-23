# # Unity ML-Agents Toolkit
import yaml

import os
import numpy as np
import json

from typing import Callable, Optional, List

import mlagents.trainers
import mlagents_envs
from mlagents import tf_utils
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.trainer_util import TrainerFactory, handle_existing_directories
from mlagents.trainers.stats import (
    TensorboardWriter,
    StatsReporter,
    GaugeWriter,
    ConsoleWriter,
)
from mlagents.trainers.cli_utils import parser, DetectDefault
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.settings import RunOptions

from mlagents.trainers.training_status import GlobalTrainingStatus
from mlagents_envs.base_env import BaseEnv
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig
from mlagents_envs.timers import (
    hierarchical_timer,
    get_timer_tree,
    add_metadata as add_timer_metadata,
)
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)

TRAINING_STATUS_FILE_NAME = "training_status.json"


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
        checkpoint_settings = options.checkpoint_settings
        env_settings = options.env_settings
        engine_settings = options.engine_settings
        base_path = "results"
        write_path = os.path.join(base_path, checkpoint_settings.run_id)
        maybe_init_path = (
            os.path.join(base_path, checkpoint_settings.initialize_from)
            if checkpoint_settings.initialize_from is not None
            else None
        )
        run_logs_dir = os.path.join(write_path, "run_logs")
        port: Optional[int] = env_settings.base_port
        # Check if directory exists
        handle_existing_directories(
            write_path,
            checkpoint_settings.resume,
            checkpoint_settings.force,
            maybe_init_path,
        )
        # Make run logs directory
        os.makedirs(run_logs_dir, exist_ok=True)
        # Load any needed states
        if checkpoint_settings.resume:
            GlobalTrainingStatus.load_state(
                os.path.join(run_logs_dir, "training_status.json")
            )

        # Configure Tensorboard Writers and StatsReporter
        tb_writer = TensorboardWriter(
            write_path, clear_past_data=not checkpoint_settings.resume
        )
        gauge_write = GaugeWriter()
        console_writer = ConsoleWriter()
        StatsReporter.add_writer(tb_writer)
        StatsReporter.add_writer(gauge_write)
        StatsReporter.add_writer(console_writer)

        if env_settings.env_path is None:
            port = None
        env_factory = create_environment_factory(
            env_settings.env_path,
            engine_settings.no_graphics,
            run_seed,
            port,
            env_settings.env_args,
            os.path.abspath(run_logs_dir),  # Unity environment requires absolute path
        )
        engine_config = EngineConfig(
            width=engine_settings.width,
            height=engine_settings.height,
            quality_level=engine_settings.quality_level,
            time_scale=engine_settings.time_scale,
            target_frame_rate=engine_settings.target_frame_rate,
            capture_frame_rate=engine_settings.capture_frame_rate,
        )
        env_manager = SubprocessEnvManager(
            env_factory, engine_config, env_settings.num_envs
        )
        env_parameter_manager = EnvironmentParameterManager(
            options.environment_parameters, run_seed, restore=checkpoint_settings.resume
        )

        trainer_factory = TrainerFactory(
            trainer_config=options.behaviors,
            output_path=write_path,
            train_model=not checkpoint_settings.inference,
            load_model=checkpoint_settings.resume,
            seed=run_seed,
            param_manager=env_parameter_manager,
            init_path=maybe_init_path,
            multi_gpu=False,
            force_torch="torch" in DetectDefault.non_default_args,
        )
        # Create controller and begin training.
        tc = TrainerController(
            trainer_factory,
            write_path,
            checkpoint_settings.run_id,
            env_parameter_manager,
            not checkpoint_settings.inference,
            run_seed,
        )

    # Begin training
    try:
        tc.start_learning(env_manager)
    finally:
        env_manager.close()
        write_run_options(write_path, options)
        write_timing_tree(run_logs_dir)
        write_training_status(run_logs_dir)


def write_run_options(output_dir: str, run_options: RunOptions) -> None:
    run_options_path = os.path.join(output_dir, "configuration.yaml")
    try:
        with open(run_options_path, "w") as f:
            try:
                yaml.dump(run_options.as_dict(), f, sort_keys=False)
            except TypeError:  # Older versions of pyyaml don't support sort_keys
                yaml.dump(run_options.as_dict(), f)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save configuration to {run_options_path}. Make sure the directory exists"
        )


def write_training_status(output_dir: str) -> None:
    GlobalTrainingStatus.save_state(os.path.join(output_dir, TRAINING_STATUS_FILE_NAME))


def write_timing_tree(output_dir: str) -> None:
    timing_path = os.path.join(output_dir, "timers.json")
    try:
        with open(timing_path, "w") as f:
            json.dump(get_timer_tree(), f, indent=4)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save to {timing_path}. Make sure the directory exists"
        )


def create_environment_factory(
    env_path: Optional[str],
    no_graphics: bool,
    seed: int,
    start_port: Optional[int],
    env_args: Optional[List[str]],
    log_folder: str,
) -> Callable[[int, List[SideChannel]], BaseEnv]:
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
            additional_args=env_args,
            side_channels=side_channels,
            log_folder=log_folder,
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
    logger.debug(json.dumps(options.as_dict(), indent=4))

    # Options deprecation warnings
    if options.checkpoint_settings.load_model:
        logger.warning(
            "The --load option has been deprecated. Please use the --resume option instead."
        )
    if options.checkpoint_settings.train_model:
        logger.warning(
            "The --train option has been deprecated. Train mode is now the default. Use "
            "--inference to run in inference mode."
        )

    run_seed = options.env_settings.seed

    # Add some timer metadata
    add_timer_metadata("mlagents_version", mlagents.trainers.__version__)
    add_timer_metadata("mlagents_envs_version", mlagents_envs.__version__)
    add_timer_metadata("communication_protocol_version", UnityEnvironment.API_VERSION)
    add_timer_metadata("tensorflow_version", tf_utils.tf.__version__)
    add_timer_metadata("numpy_version", np.__version__)

    if options.env_settings.seed == -1:
        run_seed = np.random.randint(0, 10000)
        logger.info(f"run_seed set to {run_seed}")
    run_training(run_seed, options)


def main():
    run_cli(parse_command_line())


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
