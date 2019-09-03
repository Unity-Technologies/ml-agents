from typing import Any, Dict

from mlagents.trainers import MetaCurriculum
from mlagents.envs.exception import UnityEnvironmentException
from mlagents.trainers import Trainer
from mlagents.envs.brain import BrainParameters
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.bc.offline_trainer import OfflineBCTrainer
from mlagents.trainers.bc.online_trainer import OnlineBCTrainer


def initialize_trainers(
    trainer_config: Dict[str, Any],
    external_brains: Dict[str, BrainParameters],
    summaries_dir: str,
    run_id: str,
    model_path: str,
    keep_checkpoints: int,
    train_model: bool,
    load_model: bool,
    seed: int,
    meta_curriculum: MetaCurriculum = None,
    multi_gpu: bool = False,
) -> Dict[str, Trainer]:
    """
    Initializes trainers given a provided trainer configuration and set of brains from the environment, as well as
    some general training session options.

    :param trainer_config: Original trainer configuration loaded from YAML
    :param external_brains: BrainParameters provided by the Unity environment
    :param summaries_dir: Directory to store trainer summary statistics
    :param run_id: Run ID to associate with this training run
    :param model_path: Path to save the model
    :param keep_checkpoints: How many model checkpoints to keep
    :param train_model: Whether to train the model (vs. run inference)
    :param load_model: Whether to load the model or randomly initialize
    :param seed: The random seed to use
    :param meta_curriculum: Optional meta_curriculum, used to determine a reward buffer length for PPOTrainer
    :param multi_gpu: Whether to use multi-GPU training
    :return:
    """
    trainers = {}
    trainer_parameters_dict = {}
    for brain_name in external_brains:
        trainer_parameters = trainer_config["default"].copy()
        trainer_parameters["summary_path"] = "{basedir}/{name}".format(
            basedir=summaries_dir, name=str(run_id) + "_" + brain_name
        )
        trainer_parameters["model_path"] = "{basedir}/{name}".format(
            basedir=model_path, name=brain_name
        )
        trainer_parameters["keep_checkpoints"] = keep_checkpoints
        if brain_name in trainer_config:
            _brain_key: Any = brain_name
            while not isinstance(trainer_config[_brain_key], dict):
                _brain_key = trainer_config[_brain_key]
            trainer_parameters.update(trainer_config[_brain_key])
        trainer_parameters_dict[brain_name] = trainer_parameters.copy()
    for brain_name in external_brains:
        if trainer_parameters_dict[brain_name]["trainer"] == "offline_bc":
            trainers[brain_name] = OfflineBCTrainer(
                external_brains[brain_name],
                trainer_parameters_dict[brain_name],
                train_model,
                load_model,
                seed,
                run_id,
            )
        elif trainer_parameters_dict[brain_name]["trainer"] == "online_bc":
            trainers[brain_name] = OnlineBCTrainer(
                external_brains[brain_name],
                trainer_parameters_dict[brain_name],
                train_model,
                load_model,
                seed,
                run_id,
            )
        elif trainer_parameters_dict[brain_name]["trainer"] == "ppo":
            trainers[brain_name] = PPOTrainer(
                external_brains[brain_name],
                meta_curriculum.brains_to_curriculums[brain_name].min_lesson_length
                if meta_curriculum
                else 1,
                trainer_parameters_dict[brain_name],
                train_model,
                load_model,
                seed,
                run_id,
                multi_gpu,
            )
        else:
            raise UnityEnvironmentException(
                "The trainer config contains "
                "an unknown trainer type for "
                "brain {}".format(brain_name)
            )
    return trainers
