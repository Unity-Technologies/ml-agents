# # Unity ML-Agents Toolkit

import logging

from multiprocessing import Process, Queue
import os
import glob
import shutil
import numpy as np
import yaml
from docopt import docopt
from typing import Any, Callable, Dict, Optional, NamedTuple


from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.exception import TrainerError
from mlagents.trainers import MetaCurriculumError, MetaCurriculum
from mlagents.envs import UnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.subprocess_env_manager import SubprocessEnvManager
from mlagents.envs.brain import BrainParameters


class RunArgs(NamedTuple):
    docker_target_name: Optional[str]
    env_path: Optional[str]
    run_id: str
    load_model: bool
    train_model: bool
    save_freq: int
    keep_checkpoints: int
    base_port: int
    num_envs: int
    curriculum_folder: Optional[str]
    lesson: int
    fast_simulation: bool
    no_graphics: bool
    trainer_config_path: str
    model_path: str
    summaries_dir: str

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise UnityEnvironmentException(
                "The folder {} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly.".format(model_path)
            )

    @staticmethod
    def from_docopt_dict(docopt_dict: Dict[str, Any], sub_id: int) -> "RunArgs":
        # Docker Parameters
        docker_target_name = (
            docopt_dict["--docker-target-name"]
            if docopt_dict["--docker-target-name"] != "None"
            else None
        )

        # General parameters
        env_path = docopt_dict["--env"] if docopt_dict["--env"] != "None" else None
        run_id = docopt_dict["--run-id"]
        load_model = docopt_dict["--load"]
        train_model = docopt_dict["--train"]
        save_freq = int(docopt_dict["--save-freq"])
        keep_checkpoints = int(docopt_dict["--keep-checkpoints"])
        base_port = int(docopt_dict["--base-port"])
        num_envs = int(docopt_dict["--num-envs"])
        curriculum_folder = (
            docopt_dict["--curriculum"]
            if docopt_dict["--curriculum"] != "None"
            else None
        )
        lesson = int(docopt_dict["--lesson"])
        fast_simulation = not bool(docopt_dict["--slow"])
        no_graphics = docopt_dict["--no-graphics"]
        trainer_config_path = docopt_dict["<trainer-config-path>"]

        # Recognize and use docker volume if one is passed as an argument
        if not docker_target_name:
            model_path = "./models/{run_id}-{sub_id}".format(
                run_id=run_id, sub_id=sub_id
            )
            summaries_dir = "./summaries"
        else:
            trainer_config_path = "/{docker_target_name}/{trainer_config_path}".format(
                docker_target_name=docker_target_name,
                trainer_config_path=trainer_config_path,
            )
            if curriculum_folder is not None:
                curriculum_folder = "/{docker_target_name}/{curriculum_folder}".format(
                    docker_target_name=docker_target_name,
                    curriculum_folder=curriculum_folder,
                )
            model_path = "/{docker_target_name}/models/{run_id}-{sub_id}".format(
                docker_target_name=docker_target_name, run_id=run_id, sub_id=sub_id
            )
            summaries_dir = "/{docker_target_name}/summaries".format(
                docker_target_name=docker_target_name
            )
        RunArgs._create_model_path(model_path)
        return RunArgs(
            docker_target_name,
            env_path,
            run_id,
            load_model,
            train_model,
            save_freq,
            keep_checkpoints,
            base_port,
            num_envs,
            curriculum_folder,
            lesson,
            fast_simulation,
            no_graphics,
            trainer_config_path,
            model_path,
            summaries_dir,
        )


def initialize_trainers(
    run_id: str,
    summaries_dir: str,
    model_path: str,
    keep_checkpoints: int,
    train_model: bool,
    load_model: bool,
    seed: int,
    external_brains: Dict[str, BrainParameters],
    trainer_config: Dict[str, Any],
    meta_curriculum: Optional[MetaCurriculum],
) -> Dict[str, Trainer]:
    """
    Initialization of the trainers
    :param trainer_config: The configurations of the trainers
    """
    trainer_parameters_dict = {}
    trainers = {}
    trainer_config = trainer_config
    for brain_name in external_brains:
        trainer_parameters = trainer_config["default"].copy()
        trainer_parameters["summary_path"] = "{basedir}/{name}".format(
            basedir=summaries_dir, name=str(run_id) + "_" + brain_name
        )
        trainer_parameters["model_path"] = "{basedir}/{name}".format(
            basedir=model_path, name=brain_name
        )
        trainer_parameters["keep_checkpoints"] = keep_checkpoints
        trainer_parameters["training"] = train_model
        trainer_parameters["load"] = load_model
        trainer_parameters["seed"] = seed
        trainer_parameters["run_id"] = run_id
        trainer_parameters["reward_buff_cap"] = (
            meta_curriculum.brains_to_curriculums[brain_name].min_lesson_length
            if meta_curriculum
            else 1
        )
        if brain_name in trainer_config:
            _brain_key: Any = brain_name
            while not isinstance(trainer_config[_brain_key], dict):
                _brain_key = trainer_config[_brain_key]
            trainer_parameters.update(trainer_config[_brain_key])
        trainer_parameters_dict[brain_name] = trainer_parameters.copy()
    for brain_name in external_brains:
        trainer = Trainer.initialize_from_trainer_config(
            trainer_parameters_dict[brain_name], external_brains[brain_name]
        )
        if trainer is None:
            raise UnityEnvironmentException(
                "The trainer config contains "
                "an unknown trainer type for "
                "brain {}".format(brain_name)
            )
        else:
            trainers[brain_name] = trainer
    return trainers


def run_training(
    sub_id: int, run_seed: int, run_options: Dict[str, Any], process_queue: Queue
) -> None:
    """
    Launches training session.
    :param process_queue: Queue used to send signal back to main.
    :param sub_id: Unique id for training session.
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    run_args = RunArgs.from_docopt_dict(run_options, sub_id)
    trainer_config = load_config(run_args.trainer_config_path)
    env_factory = create_environment_factory(
        run_args.env_path,
        run_args.docker_target_name,
        run_args.no_graphics,
        run_seed,
        run_args.base_port + (sub_id * run_args.num_envs),
    )
    env = SubprocessEnvManager(env_factory, run_args.num_envs)
    meta_curriculum = None
    if run_args.curriculum_folder is not None:
        meta_curriculum = create_meta_curriculum(run_args.curriculum_folder, env)
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        meta_curriculum.set_all_curriculums_to_lesson_num(run_args.lesson)

    trainers = initialize_trainers(
        run_args.run_id,
        run_args.summaries_dir,
        run_args.model_path,
        run_args.keep_checkpoints,
        run_args.train_model,
        run_args.load_model,
        run_seed,
        env.external_brains,
        trainer_config,
        meta_curriculum,
    )
    # Create controller and begin training.
    tc = TrainerController(
        trainers,
        run_args.summaries_dir,
        run_args.run_id + "-" + str(sub_id),
        run_args.save_freq,
        meta_curriculum,
        run_args.train_model,
        run_args.keep_checkpoints,
        run_args.lesson,
        run_seed,
        run_args.fast_simulation,
    )

    # Signal that environment has been launched.
    process_queue.put(True)

    # Begin training
    tc.start_learning(env)


def create_meta_curriculum(
    curriculum_folder: str, env: SubprocessEnvManager
) -> MetaCurriculum:
    meta_curriculum = MetaCurriculum(curriculum_folder, env.reset_parameters)
    for brain_name in meta_curriculum.brains_to_curriculums.keys():
        if brain_name not in env.external_brains.keys():
            raise MetaCurriculumError(
                "One of the curricula "
                "defined in " + curriculum_folder + " "
                "does not have a corresponding "
                "Brain. Check that the "
                "curriculum file has the same "
                "name as the Brain "
                "whose curriculum it defines."
            )
    return meta_curriculum


def prepare_for_docker_run(docker_target_name: str, env_path: str) -> str:
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


def load_config(trainer_config_path: str) -> Dict[str, Any]:
    try:
        with open(trainer_config_path) as data_file:
            trainer_config = yaml.safe_load(data_file)
            return trainer_config
    except IOError:
        raise UnityEnvironmentException(
            "Parameter file could not be found " "at {}.".format(trainer_config_path)
        )
    except UnicodeDecodeError:
        raise UnityEnvironmentException(
            "There was an error decoding "
            "Trainer Config from this path : {}".format(trainer_config_path)
        )


def create_environment_factory(
    env_path: Optional[str],
    docker_target_name: Optional[str],
    no_graphics: bool,
    seed: Optional[int],
    start_port: int,
) -> Callable[[int], BaseUnityEnvironment]:
    if env_path is not None:
        # Strip out executable extensions if passed
        env_path = (
            env_path.strip()
            .replace(".app", "")
            .replace(".exe", "")
            .replace(".x86_64", "")
            .replace(".x86", "")
        )
    docker_training: bool = docker_target_name is not None
    if docker_target_name is not None and env_path is not None:
        """
            Comments for future maintenance:
                Some OS/VM instances (e.g. COS GCP Image) mount filesystems
                with COS flag which prevents execution of the Unity scene,
                to get around this, we will copy the executable into the
                container.
            """
        # Navigate in docker path and find env_path and copy it.
        env_path = prepare_for_docker_run(docker_target_name, env_path)
    seed_count = 10000
    seed_pool = [np.random.randint(0, seed_count) for _ in range(seed_count)]

    def create_unity_environment(worker_id: int) -> UnityEnvironment:
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

    _USAGE = """
    Usage:
      mlagents-learn <trainer-config-path> [options]
      mlagents-learn --help

    Options:
      --env=<file>               Name of the Unity executable [default: None].
      --curriculum=<directory>   Curriculum json directory for environment [default: None].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-id=<path>            The directory name for model and summary statistics [default: ppo].
      --num-runs=<n>             Number of concurrent training sessions [default: 1].
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --train                    Whether to train model, or only run inference [default: False].
      --base-port=<n>            Base port for environment communication [default: 5005].
      --num-envs=<n>             Number of parallel environments to use for training [default: 1]
      --docker-target-name=<dt>  Docker volume to store training-specific files [default: None].
      --no-graphics              Whether to run the environment in no-graphics mode [default: False].
      --debug                    Whether to run ML-Agents in debug mode with detailed logging [default: False].
    """

    options = docopt(_USAGE)
    trainer_logger = logging.getLogger("mlagents.trainers")
    env_logger = logging.getLogger("mlagents.envs")
    trainer_logger.info(options)
    if options["--debug"]:
        trainer_logger.setLevel("DEBUG")
        env_logger.setLevel("DEBUG")
    num_runs = int(options["--num-runs"])
    seed = int(options["--seed"])

    if options["--env"] == "None" and num_runs > 1:
        raise TrainerError(
            "It is not possible to launch more than one concurrent training session "
            "when training from the editor."
        )

    jobs = []
    run_seed = seed

    if num_runs == 1:
        if seed == -1:
            run_seed = np.random.randint(0, 10000)
        run_training(0, run_seed, options, Queue())
    else:
        for i in range(num_runs):
            if seed == -1:
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
