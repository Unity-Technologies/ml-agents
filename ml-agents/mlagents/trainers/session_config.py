import os
import glob
import shutil
import logging
from typing import Any, Dict, Optional, NamedTuple
from mlagents.envs.exception import UnityEnvironmentException


class SessionConfig(NamedTuple):
    docker_training: bool
    env_path: Optional[str]
    run_id: str
    sub_run_id: str
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
    def from_docopt_dict(docopt_dict: Dict[str, Any], sub_id: int) -> "SessionConfig":
        # Docker Parameters
        docker_target_name = (
            docopt_dict["--docker-target-name"]
            if docopt_dict["--docker-target-name"] != "None"
            else None
        )

        # General parameters
        env_path = docopt_dict["--env"] if docopt_dict["--env"] != "None" else None
        if env_path is not None:
            # Strip out executable extensions if passed
            env_path = (
                env_path.strip()
                .replace(".app", "")
                .replace(".exe", "")
                .replace(".x86_64", "")
                .replace(".x86", "")
            )
        run_id = docopt_dict["--run-id"]
        sub_run_id = f"{run_id}-{sub_id}"
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

        docker_training: bool = docker_target_name is not None

        # Recognize and use docker volume if one is passed as an argument
        if docker_training:
            trainer_config_path = "/{docker_target_name}/{trainer_config_path}".format(
                docker_target_name=docker_target_name,
                trainer_config_path=trainer_config_path,
            )
            if curriculum_folder is not None:
                curriculum_folder = "/{docker_target_name}/{curriculum_folder}".format(
                    docker_target_name=docker_target_name,
                    curriculum_folder=curriculum_folder,
                )
            model_path = f"/{docker_target_name}/models/{sub_run_id}"
            summaries_dir = "/{docker_target_name}/summaries".format(
                docker_target_name=docker_target_name
            )
            if env_path is not None:
                """
                    Comments for future maintenance:
                        Some OS/VM instances (e.g. COS GCP Image) mount filesystems
                        with COS flag which prevents execution of the Unity scene,
                        to get around this, we will copy the executable into the
                        container.
                """
                # Navigate in docker path and find env_path and copy it.
                env_path = prepare_for_docker_run(docker_target_name, env_path)
        else:
            model_path = f"./models/{sub_run_id}"
            summaries_dir = "./summaries"

        SessionConfig._create_model_path(model_path)
        return SessionConfig(
            docker_training,
            env_path,
            run_id,
            sub_run_id,
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
