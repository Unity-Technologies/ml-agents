import os
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.model_saver.torch_model_saver import DEFAULT_CHECKPOINT_NAME


def validate_existing_directories(
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
    :param init_path: Path to run-id dir to initialize from
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


def setup_init_path(
    behaviors: TrainerSettings.DefaultTrainerDict, init_dir: str
) -> None:
    """
    For each behavior, setup full init_path to checkpoint file to initialize policy from
    :param behaviors: mapping from behavior_name to TrainerSettings
    :param init_dir: Path to run-id dir to initialize from
    """
    for behavior_name, ts in behaviors.items():
        if ts.init_path is None:
            # set default if None
            ts.init_path = os.path.join(
                init_dir, behavior_name, DEFAULT_CHECKPOINT_NAME
            )
        elif not os.path.dirname(ts.init_path):
            # update to full path if just the file name
            ts.init_path = os.path.join(init_dir, behavior_name, ts.init_path)
        _validate_init_full_path(ts.init_path)


def _validate_init_full_path(init_file: str) -> None:
    """
    Validate initialization path to be a .pt file
    :param init_file: full path to initialization checkpoint file
    """
    if not (os.path.isfile(init_file) and init_file.endswith(".pt")):
        raise UnityTrainerException(
            f"Could not initialize from {init_file}. file does not exists or is not a `.pt` file"
        )
