import os
import argparse

from huggingface_hub import snapshot_download

from mlagents_envs import logging_util
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)


def load_from_hf(repo_id: str, local_dir: str) -> None:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param local_dir: local destination of the repository
    """
    _, repo_name = repo_id.split("/")

    local_dir = os.path.join(local_dir, repo_name)

    snapshot_download(repo_id=repo_id, local_dir=local_dir)

    logger.info(f"The repository {repo_id} has been downloaded to {local_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        help="Repo id of the model repository from the Hugging Face Hub",
        type=str,
    )
    parser.add_argument(
        "--local-dir",
        help="Local destination of the repository",
        type=str,
        default="./",
    )
    args = parser.parse_args()

    # Load model from the Hub
    load_from_hf(args.repo_id, args.local_dir)


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
