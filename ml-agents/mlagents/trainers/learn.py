# # Unity ML-Agents Toolkit
import logging
import argparse

from multiprocessing import Process, Queue
import os
import glob
import shutil
import numpy as np

from typing import Any, Callable, Optional, List, NamedTuple

import mlagents.trainers
import mlagents_envs
from mlagents import tf_utils
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.exception import TrainerError
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.trainer_util import load_config, TrainerFactory
from mlagents.trainers.stats import TensorboardWriter, CSVWriter, StatsReporter
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.exception import SamplerException
from mlagents_envs.base_env import BaseEnv
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig


class CommandLineOptions(NamedTuple):
    debug: bool
    num_runs: int
    seed: int
    env_path: str
    run_id: str
    load_model: bool
    train_model: bool
    save_freq: int
    keep_checkpoints: int
    base_port: int
    num_envs: int
    curriculum_folder: Optional[str]
    lesson: int
    no_graphics: bool
    multi_gpu: bool  # ?
    trainer_config_path: str
    sampler_file_path: Optional[str]
    docker_target_name: Optional[str]
    env_args: Optional[List[str]]
    cpu: bool
    width: int
    height: int
    quality_level: int
    time_scale: float
    target_frame_rate: int

    @staticmethod
    def from_argparse(args: Any) -> "CommandLineOptions":
        return CommandLineOptions(**vars(args))


def get_version_string() -> str:
    # pylint: disable=no-member
    return f""" Version information:
  ml-agents: {mlagents.trainers.__version__},
  ml-agents-envs: {mlagents_envs.__version__},
  Communicator API: {UnityEnvironment.API_VERSION},
  TensorFlow: {tf_utils.tf.__version__}"""


def parse_command_line(argv: Optional[List[str]] = None) -> CommandLineOptions:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("trainer_config_path")
    parser.add_argument(
        "--env", default=None, dest="env_path", help="Name of the Unity executable "
    )
    parser.add_argument(
        "--curriculum",
        default=None,
        dest="curriculum_folder",
        help="Curriculum json directory for environment",
    )
    parser.add_argument(
        "--sampler",
        default=None,
        dest="sampler_file_path",
        help="Reset parameter yaml file for environment",
    )
    parser.add_argument(
        "--keep-checkpoints",
        default=5,
        type=int,
        help="How many model checkpoints to keep",
    )
    parser.add_argument(
        "--lesson", default=0, type=int, help="Start learning from this lesson"
    )
    parser.add_argument(
        "--load",
        default=False,
        dest="load_model",
        action="store_true",
        help="Whether to load the model or randomly initialize",
    )
    parser.add_argument(
        "--run-id",
        default="ppo",
        help="The directory name for model and summary statistics",
    )
    parser.add_argument(
        "--num-runs", default=1, type=int, help="Number of concurrent training sessions"
    )
    parser.add_argument(
        "--save-freq", default=50000, type=int, help="Frequency at which to save model"
    )
    parser.add_argument(
        "--seed", default=-1, type=int, help="Random seed used for training"
    )
    parser.add_argument(
        "--train",
        default=False,
        dest="train_model",
        action="store_true",
        help="Whether to train model, or only run inference",
    )
    parser.add_argument(
        "--base-port",
        default=5005,
        type=int,
        help="Base port for environment communication",
    )
    parser.add_argument(
        "--num-envs",
        default=1,
        type=int,
        help="Number of parallel environments to use for training",
    )
    parser.add_argument(
        "--docker-target-name",
        default=None,
        dest="docker_target_name",
        help="Docker volume to store training-specific files",
    )
    parser.add_argument(
        "--no-graphics",
        default=False,
        action="store_true",
        help="Whether to run the environment in no-graphics mode",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to run ML-Agents in debug mode with detailed logging",
    )
    parser.add_argument(
        "--multi-gpu",
        default=False,
        action="store_true",
        help="Setting this flag enables the use of multiple GPU's (if available) during training",
    )
    parser.add_argument(
        "--env-args",
        default=None,
        nargs=argparse.REMAINDER,
        help="Arguments passed to the Unity executable.",
    )
    parser.add_argument(
        "--cpu", default=False, action="store_true", help="Run with CPU only"
    )

    parser.add_argument("--version", action="version", version="")

    eng_conf = parser.add_argument_group(title="Engine Configuration")
    eng_conf.add_argument(
        "--width",
        default=84,
        type=int,
        help="The width of the executable window of the environment(s)",
    )
    eng_conf.add_argument(
        "--height",
        default=84,
        type=int,
        help="The height of the executable window of the environment(s)",
    )
    eng_conf.add_argument(
        "--quality-level",
        default=5,
        type=int,
        help="The quality level of the environment(s)",
    )
    eng_conf.add_argument(
        "--time-scale",
        default=20,
        type=float,
        help="The time scale of the Unity environment(s)",
    )
    eng_conf.add_argument(
        "--target-frame-rate",
        default=-1,
        type=int,
        help="The target frame rate of the Unity environment(s)",
    )

    args = parser.parse_args(argv)
    return CommandLineOptions.from_argparse(args)


def run_training(
    sub_id: int, run_seed: int, options: CommandLineOptions, process_queue: Queue
) -> None:
    """
    Launches training session.
    :param process_queue: Queue used to send signal back to main.
    :param sub_id: Unique id for training session.
    :param options: parsed command line arguments
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    # Docker Parameters
    trainer_config_path = options.trainer_config_path
    curriculum_folder = options.curriculum_folder
    # Recognize and use docker volume if one is passed as an argument
    if not options.docker_target_name:
        model_path = "./models/{run_id}-{sub_id}".format(
            run_id=options.run_id, sub_id=sub_id
        )
        summaries_dir = "./summaries"
    else:
        trainer_config_path = "/{docker_target_name}/{trainer_config_path}".format(
            docker_target_name=options.docker_target_name,
            trainer_config_path=trainer_config_path,
        )
        if curriculum_folder is not None:
            curriculum_folder = "/{docker_target_name}/{curriculum_folder}".format(
                docker_target_name=options.docker_target_name,
                curriculum_folder=curriculum_folder,
            )
        model_path = "/{docker_target_name}/models/{run_id}-{sub_id}".format(
            docker_target_name=options.docker_target_name,
            run_id=options.run_id,
            sub_id=sub_id,
        )
        summaries_dir = "/{docker_target_name}/summaries".format(
            docker_target_name=options.docker_target_name
        )
    trainer_config = load_config(trainer_config_path)
    port = options.base_port + (sub_id * options.num_envs)

    # Configure CSV, Tensorboard Writers and StatsReporter
    # We assume reward and episode length are needed in the CSV.
    csv_writer = CSVWriter(
        summaries_dir,
        required_fields=["Environment/Cumulative Reward", "Environment/Episode Length"],
    )
    tb_writer = TensorboardWriter(summaries_dir)
    StatsReporter.add_writer(tb_writer)
    StatsReporter.add_writer(csv_writer)

    if options.env_path is None:
        port = 5004  # This is the in Editor Training Port
    env_factory = create_environment_factory(
        options.env_path,
        options.docker_target_name,
        options.no_graphics,
        run_seed,
        port,
        options.env_args,
    )
    engine_config = EngineConfig(
        options.width,
        options.height,
        options.quality_level,
        options.time_scale,
        options.target_frame_rate,
    )
    env_manager = SubprocessEnvManager(env_factory, engine_config, options.num_envs)
    maybe_meta_curriculum = try_create_meta_curriculum(
        curriculum_folder, env_manager, options.lesson
    )
    sampler_manager, resampling_interval = create_sampler_manager(
        options.sampler_file_path, run_seed
    )
    trainer_factory = TrainerFactory(
        trainer_config,
        summaries_dir,
        options.run_id,
        model_path,
        options.keep_checkpoints,
        options.train_model,
        options.load_model,
        run_seed,
        maybe_meta_curriculum,
        options.multi_gpu,
    )
    # Create controller and begin training.
    tc = TrainerController(
        trainer_factory,
        model_path,
        summaries_dir,
        options.run_id + "-" + str(sub_id),
        options.save_freq,
        maybe_meta_curriculum,
        options.train_model,
        run_seed,
        sampler_manager,
        resampling_interval,
    )
    # Signal that environment has been launched.
    process_queue.put(True)
    # Begin training
    try:
        tc.start_learning(env_manager)
    finally:
        env_manager.close()


def create_sampler_manager(sampler_file_path, run_seed=None):
    sampler_config = None
    resample_interval = None
    if sampler_file_path is not None:
        sampler_config = load_config(sampler_file_path)
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
    curriculum_folder: Optional[str], env: SubprocessEnvManager, lesson: int
) -> Optional[MetaCurriculum]:
    if curriculum_folder is None:
        return None

    else:
        meta_curriculum = MetaCurriculum(curriculum_folder)
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        meta_curriculum.set_all_curriculums_to_lesson_num(lesson)

        return meta_curriculum


def prepare_for_docker_run(docker_target_name, env_path):
    for f in glob.glob(
        "/{docker_target_name}/*".format(docker_target_name=docker_target_name)
    ):
        if env_path in f:
            try:
                b = os.path.basename(f)
                if os.path.isdir(f):
                    shutil.copytree(f, "/ml-agents/{b}".format(b=b))
                else:
                    src_f = "/{docker_target_name}/{b}".format(
                        docker_target_name=docker_target_name, b=b
                    )
                    dst_f = "/ml-agents/{b}".format(b=b)
                    shutil.copyfile(src_f, dst_f)
                    os.chmod(dst_f, 0o775)  # Make executable
            except Exception as e:
                logging.getLogger("mlagents.trainers").info(e)
    env_path = "/ml-agents/{env_path}".format(env_path=env_path)
    return env_path


def create_environment_factory(
    env_path: str,
    docker_target_name: Optional[str],
    no_graphics: bool,
    seed: Optional[int],
    start_port: int,
    env_args: Optional[List[str]],
) -> Callable[[int, List[SideChannel]], BaseEnv]:
    if env_path is not None:
        # Strip out executable extensions if passed
        env_path = (
            env_path.strip()
            .replace(".app", "")
            .replace(".exe", "")
            .replace(".x86_64", "")
            .replace(".x86", "")
        )
    docker_training = docker_target_name is not None
    if docker_training and env_path is not None:
        #     Comments for future maintenance:
        #         Some OS/VM instances (e.g. COS GCP Image) mount filesystems
        #         with COS flag which prevents execution of the Unity scene,
        #         to get around this, we will copy the executable into the
        #         container.
        # Navigate in docker path and find env_path and copy it.
        env_path = prepare_for_docker_run(docker_target_name, env_path)
    seed_count = 10000
    seed_pool = [np.random.randint(0, seed_count) for _ in range(seed_count)]

    def create_unity_environment(
        worker_id: int, side_channels: List[SideChannel]
    ) -> UnityEnvironment:
        env_seed = seed
        if not env_seed:
            env_seed = seed_pool[worker_id % len(seed_pool)]
        return UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            seed=env_seed,
            docker_training=docker_training,
            no_graphics=no_graphics,
            base_port=start_port,
            args=env_args,
            side_channels=side_channels,
        )

    return create_unity_environment


def main():
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
    options = parse_command_line()
    trainer_logger = logging.getLogger("mlagents.trainers")
    env_logger = logging.getLogger("mlagents_envs")
    trainer_logger.info(options)
    if options.debug:
        trainer_logger.setLevel("DEBUG")
        env_logger.setLevel("DEBUG")
    else:
        # disable noisy warnings from tensorflow.
        tf_utils.set_warnings_enabled(False)
    if options.env_path is None and options.num_runs > 1:
        raise TrainerError(
            "It is not possible to launch more than one concurrent training session "
            "when training from the editor."
        )

    jobs = []
    run_seed = options.seed
    if options.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if options.num_runs == 1:
        if options.seed == -1:
            run_seed = np.random.randint(0, 10000)
        run_training(0, run_seed, options, Queue())
    else:
        for i in range(options.num_runs):
            if options.seed == -1:
                run_seed = np.random.randint(0, 10000)
            process_queue = Queue()
            p = Process(target=run_training, args=(i, run_seed, options, process_queue))
            jobs.append(p)
            p.start()
            # Wait for signal that environment has successfully launched
            while process_queue.get() is not True:
                continue

    # Wait for jobs to complete.  Otherwise we'll have an extra
    # unhandled KeyboardInterrupt if we end early.
    try:
        for job in jobs:
            job.join()
    except KeyboardInterrupt:
        pass


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
