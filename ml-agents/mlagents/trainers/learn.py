# # Unity ML-Agents Toolkit

import logging

from multiprocessing import Process, Queue
import os
import glob
import shutil
import numpy as np
import yaml
from docopt import docopt
from typing import Optional, Callable


from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.exception import TrainerError
from mlagents.trainers import MetaCurriculumError, MetaCurriculum
from mlagents.envs import UnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.subprocess_environment import SubprocessUnityEnvironment


def run_training(sub_id: int, run_seed: int, run_options, process_queue):
    """
    Launches training session.
    :param process_queue: Queue used to send signal back to main.
    :param sub_id: Unique id for training session.
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    # Docker Parameters
    docker_target_name = (
        run_options["--docker-target-name"]
        if run_options["--docker-target-name"] != "None"
        else None
    )

    # General parameters
    env_path = run_options["--env"] if run_options["--env"] != "None" else None
    run_id = run_options["--run-id"]
    load_model = run_options["--load"]
    train_model = run_options["--train"]
    save_freq = int(run_options["--save-freq"])
    keep_checkpoints = int(run_options["--keep-checkpoints"])
    base_port = int(run_options["--base-port"])
    num_envs = int(run_options["--num-envs"])
    curriculum_folder = (
        run_options["--curriculum"] if run_options["--curriculum"] != "None" else None
    )
    lesson = int(run_options["--lesson"])
    fast_simulation = not bool(run_options["--slow"])
    no_graphics = run_options["--no-graphics"]
    trainer_config_path = run_options["<trainer-config-path>"]
    # Recognize and use docker volume if one is passed as an argument
    if not docker_target_name:
        model_path = "./models/{run_id}-{sub_id}".format(run_id=run_id, sub_id=sub_id)
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

    trainer_config = load_config(trainer_config_path)
    env_factory = create_environment_factory(
        env_path,
        docker_target_name,
        no_graphics,
        run_seed,
        base_port + (sub_id * num_envs),
    )
    env = SubprocessUnityEnvironment(env_factory, num_envs)
    maybe_meta_curriculum = try_create_meta_curriculum(curriculum_folder, env)

    # Create controller and begin training.
    tc = TrainerController(
        model_path,
        summaries_dir,
        run_id + "-" + str(sub_id),
        save_freq,
        maybe_meta_curriculum,
        load_model,
        train_model,
        keep_checkpoints,
        lesson,
        env.external_brains,
        run_seed,
        fast_simulation,
    )

    # Signal that environment has been launched.
    process_queue.put(True)

    # Begin training
    tc.start_learning(env, trainer_config)


def try_create_meta_curriculum(
    curriculum_folder: Optional[str], env: BaseUnityEnvironment
) -> Optional[MetaCurriculum]:
    if curriculum_folder is None:
        return None
    else:
        meta_curriculum = MetaCurriculum(curriculum_folder, env.reset_parameters)
        if meta_curriculum:
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


def load_config(trainer_config_path):
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
    env_path: str,
    docker_target_name: str,
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
    docker_training = docker_target_name is not None
    if docker_training and env_path is not None:
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
    except:
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
