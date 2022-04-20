from huggingface_hub import Repository
import os

import argparse


def load_from_hf(repo_id: str, 
                 local_dir: str):
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param local_dir: local destination of the repository
    """
    temp = repo_id.split('/')
    organization = temp[0]
    repo_name = temp[1]
    print("REPO NAME: ", repo_name)
    print("ORGANIZATION: ", organization)

    local_dir = os.path.join(local_dir, repo_name)
    repo = Repository(local_dir, repo_id)
    print(f"The repository {repo_id} has been cloned to {local_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", help="Repo id of the model repository from the Hugging Face Hub", type=str)
    parser.add_argument("--local-dir", help="Local destination of the repository", type=str, default="./")
    args = parser.parse_args()

    # Load model from hub
    load_from_hf(args.repo_id, args.local_dir)


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
