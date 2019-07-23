# # Unity ML-Agents Toolkit

import logging

from multiprocessing import Process, Queue
import numpy as np
import yaml
from docopt import docopt
from typing import Any, Callable, Dict, Optional


from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.exception import TrainerError
from mlagents.trainers import MetaCurriculumError, MetaCurriculum
from mlagents.trainers.session_config import SessionConfig
from mlagents.envs import UnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.subprocess_env_manager import SubprocessEnvManager
from mlagents.envs.brain import BrainParameters


def initialize_trainers(
    sess_config: SessionConfig,
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
            basedir=sess_config.summaries_dir,
            name=str(sess_config.run_id) + "_" + brain_name,
        )
        trainer_parameters["model_path"] = "{basedir}/{name}".format(
            basedir=sess_config.model_path, name=brain_name
        )
        trainer_parameters["keep_checkpoints"] = sess_config.keep_checkpoints
        trainer_parameters["training"] = sess_config.train_model
        trainer_parameters["load"] = sess_config.load_model
        trainer_parameters["seed"] = seed
        trainer_parameters["run_id"] = sess_config.run_id
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
    sess_config = SessionConfig.from_docopt_dict(run_options, sub_id)
    trainer_config = load_config(sess_config.trainer_config_path)
    env_factory = create_environment_factory(
        sess_config, sess_config.base_port + (sub_id * sess_config.num_envs), run_seed
    )
    env = SubprocessEnvManager(env_factory, sess_config.num_envs)
    meta_curriculum = None
    if sess_config.curriculum_folder is not None:
        meta_curriculum = create_meta_curriculum(sess_config.curriculum_folder, env)
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        meta_curriculum.set_all_curriculums_to_lesson_num(sess_config.lesson)

    trainers = initialize_trainers(
        sess_config, run_seed, env.external_brains, trainer_config, meta_curriculum
    )
    # Create controller and begin training.
    tc = TrainerController(trainers, sess_config, run_seed, meta_curriculum)

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
    sess_config: SessionConfig, start_port: int, seed: int
) -> Callable[[int], BaseUnityEnvironment]:
    seed_count = 10000
    seed_pool = [np.random.randint(0, seed_count) for _ in range(seed_count)]

    def create_unity_environment(worker_id: int) -> UnityEnvironment:
        env_seed = seed
        if not env_seed:
            env_seed = seed_pool[worker_id % len(seed_pool)]
        return UnityEnvironment(
            file_name=sess_config.env_path,
            worker_id=worker_id,
            seed=env_seed,
            docker_training=sess_config.docker_training,
            no_graphics=sess_config.no_graphics,
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
