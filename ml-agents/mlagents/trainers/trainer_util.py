import yaml
from typing import Any, Dict, TextIO
import logging

from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.trainer import Trainer, UnityTrainerException
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer

logger = logging.getLogger("mlagents.trainers")


class TrainerFactory:
    def __init__(
        self,
        trainer_config: Any,
        summaries_dir: str,
        run_id: str,
        model_path: str,
        keep_checkpoints: int,
        train_model: bool,
        load_model: bool,
        seed: int,
        meta_curriculum: MetaCurriculum = None,
        multi_gpu: bool = False,
    ):
        self.trainer_config = trainer_config
        self.summaries_dir = summaries_dir
        self.run_id = run_id
        self.model_path = model_path
        self.keep_checkpoints = keep_checkpoints
        self.train_model = train_model
        self.load_model = load_model
        self.seed = seed
        self.meta_curriculum = meta_curriculum
        self.multi_gpu = multi_gpu

    def generate(self, brain_name: str) -> Trainer:
        return initialize_trainer(
            self.trainer_config,
            brain_name,
            self.summaries_dir,
            self.run_id,
            self.model_path,
            self.keep_checkpoints,
            self.train_model,
            self.load_model,
            self.seed,
            self.meta_curriculum,
            self.multi_gpu,
        )


def initialize_trainer(
    trainer_config: Any,
    brain_name: str,
    summaries_dir: str,
    run_id: str,
    model_path: str,
    keep_checkpoints: int,
    train_model: bool,
    load_model: bool,
    seed: int,
    meta_curriculum: MetaCurriculum = None,
    multi_gpu: bool = False,
) -> Trainer:
    """
    Initializes a trainer given a provided trainer configuration and brain parameters, as well as
    some general training session options.

    :param trainer_config: Original trainer configuration loaded from YAML
    :param brain_name: Name of the brain to be associated with trainer
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
    if "default" not in trainer_config and brain_name not in trainer_config:
        raise TrainerConfigError(
            f'Trainer config must have either a "default" section, or a section for the brain name ({brain_name}). '
            "See config/trainer_config.yaml for an example."
        )

    trainer_parameters = trainer_config.get("default", {}).copy()
    trainer_parameters["summary_path"] = str(run_id) + "_" + brain_name
    trainer_parameters["model_path"] = "{basedir}/{name}".format(
        basedir=model_path, name=brain_name
    )
    trainer_parameters["keep_checkpoints"] = keep_checkpoints
    if brain_name in trainer_config:
        _brain_key: Any = brain_name
        while not isinstance(trainer_config[_brain_key], dict):
            _brain_key = trainer_config[_brain_key]
        trainer_parameters.update(trainer_config[_brain_key])

    min_lesson_length = 1
    if meta_curriculum:
        if brain_name in meta_curriculum.brains_to_curriculums:
            min_lesson_length = meta_curriculum.brains_to_curriculums[
                brain_name
            ].min_lesson_length
        else:
            logger.warning(
                f"Metacurriculum enabled, but no curriculum for brain {brain_name}. "
                f"Brains with curricula: {meta_curriculum.brains_to_curriculums.keys()}. "
            )

    trainer: Trainer = None  # type: ignore  # will be set to one of these, or raise
    if "trainer" not in trainer_parameters:
        raise TrainerConfigError(
            f'The "trainer" key must be set in your trainer config for brain {brain_name} (or the default brain).'
        )
    trainer_type = trainer_parameters["trainer"]

    if trainer_type == "offline_bc":
        raise UnityTrainerException(
            "The offline_bc trainer has been removed. To train with demonstrations, "
            "please use a PPO or SAC trainer with the GAIL Reward Signal and/or the "
            "Behavioral Cloning feature enabled."
        )
    elif trainer_type == "ppo":
        trainer = PPOTrainer(
            brain_name,
            min_lesson_length,
            trainer_parameters,
            train_model,
            load_model,
            seed,
            run_id,
            multi_gpu,
        )
    elif trainer_type == "sac":
        trainer = SACTrainer(
            brain_name,
            min_lesson_length,
            trainer_parameters,
            train_model,
            load_model,
            seed,
            run_id,
        )
    else:
        raise TrainerConfigError(
            f'The trainer config contains an unknown trainer type "{trainer_type}" for brain {brain_name}'
        )
    return trainer


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path) as data_file:
            return _load_config(data_file)
    except IOError:
        raise TrainerConfigError(f"Config file could not be found at {config_path}.")
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
