import os
from typing import Dict

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.ghost.trainer import GhostTrainer
from mlagents.trainers.ghost.controller import GhostController
from mlagents.trainers.settings import TrainerSettings, TrainerType


logger = get_logger(__name__)


class TrainerFactory:
    def __init__(
        self,
        trainer_config: Dict[str, TrainerSettings],
        output_path: str,
        train_model: bool,
        load_model: bool,
        seed: int,
        init_path: str = None,
        meta_curriculum: MetaCurriculum = None,
        multi_gpu: bool = False,
    ):
        self.trainer_config = trainer_config
        self.output_path = output_path
        self.init_path = init_path
        self.train_model = train_model
        self.load_model = load_model
        self.seed = seed
        self.meta_curriculum = meta_curriculum
        self.multi_gpu = multi_gpu
        self.ghost_controller = GhostController()

    def generate(self, brain_name: str) -> Trainer:
        return initialize_trainer(
            self.trainer_config[brain_name],
            brain_name,
            self.output_path,
            self.train_model,
            self.load_model,
            self.ghost_controller,
            self.seed,
            self.init_path,
            self.meta_curriculum,
            self.multi_gpu,
        )


def initialize_trainer(
    trainer_settings: TrainerSettings,
    brain_name: str,
    output_path: str,
    train_model: bool,
    load_model: bool,
    ghost_controller: GhostController,
    seed: int,
    init_path: str = None,
    meta_curriculum: MetaCurriculum = None,
    multi_gpu: bool = False,
) -> Trainer:
    """
    Initializes a trainer given a provided trainer configuration and brain parameters, as well as
    some general training session options.

    :param trainer_settings: Original trainer configuration loaded from YAML
    :param brain_name: Name of the brain to be associated with trainer
    :param output_path: Path to save the model and summary statistics
    :param keep_checkpoints: How many model checkpoints to keep
    :param train_model: Whether to train the model (vs. run inference)
    :param load_model: Whether to load the model or randomly initialize
    :param ghost_controller: The object that coordinates ghost trainers
    :param seed: The random seed to use
    :param init_path: Path from which to load model, if different from model_path.
    :param meta_curriculum: Optional meta_curriculum, used to determine a reward buffer length for PPOTrainer
    :return:
    """
    trainer_artifact_path = os.path.join(output_path, brain_name)
    if init_path is not None:
        trainer_settings.init_path = os.path.join(init_path, brain_name)

    min_lesson_length = 1
    if meta_curriculum:
        if brain_name in meta_curriculum.brains_to_curricula:
            min_lesson_length = meta_curriculum.brains_to_curricula[
                brain_name
            ].min_lesson_length
        else:
            logger.warning(
                f"Metacurriculum enabled, but no curriculum for brain {brain_name}. "
                f"Brains with curricula: {meta_curriculum.brains_to_curricula.keys()}. "
            )

    trainer: Trainer = None  # type: ignore  # will be set to one of these, or raise
    trainer_type = trainer_settings.trainer_type

    if trainer_type == TrainerType.PPO:
        trainer = PPOTrainer(
            brain_name,
            min_lesson_length,
            trainer_settings,
            train_model,
            load_model,
            seed,
            trainer_artifact_path,
        )
    elif trainer_type == TrainerType.SAC:
        trainer = SACTrainer(
            brain_name,
            min_lesson_length,
            trainer_settings,
            train_model,
            load_model,
            seed,
            trainer_artifact_path,
        )
    else:
        raise TrainerConfigError(
            f'The trainer config contains an unknown trainer type "{trainer_type}" for brain {brain_name}'
        )

    if trainer_settings.self_play is not None:
        trainer = GhostTrainer(
            trainer,
            brain_name,
            ghost_controller,
            min_lesson_length,
            trainer_settings,
            train_model,
            trainer_artifact_path,
        )
    return trainer


def handle_existing_directories(
    output_path: str, resume: bool, force: bool, init_path: str = None
) -> None:
    """
    Validates that if the run_id model exists, we do not overwrite it unless --force is specified.
    Throws an exception if resume isn't specified and run_id exists. Throws an exception
    if --resume is specified and run-id was not found.
    :param model_path: The model path specified.
    :param summary_path: The summary path to be used.
    :param resume: Whether or not the --resume flag was passed.
    :param force: Whether or not the --force flag was passed.
    """

    output_path_exists = os.path.isdir(output_path)

    if output_path_exists:
        if not resume and not force:
            raise UnityTrainerException(
                "Previous data from this run ID was found. "
                "Either specify a new run ID, use --resume to resume this run, "
                "or use the --force parameter to overwrite existing data."
            )
    else:
        if resume:
            raise UnityTrainerException(
                "Previous data from this run ID was not found. "
                "Train a new run by removing the --resume flag."
            )

    # Verify init path if specified.
    if init_path is not None:
        if not os.path.isdir(init_path):
            raise UnityTrainerException(
                "Could not initialize from {}. "
                "Make sure models have already been saved with that run ID.".format(
                    init_path
                )
            )
