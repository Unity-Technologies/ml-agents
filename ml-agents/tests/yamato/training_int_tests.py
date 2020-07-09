import argparse
import os
import shutil
import sys
import subprocess
import time
from typing import Any

from .yamato_utils import (
    find_executables,
    get_base_path,
    get_base_output_path,
    run_standalone_build,
    init_venv,
    override_config_file,
    override_legacy_config_file,
    checkout_csharp_version,
    undo_git_checkout,
)


def run_training(python_version: str, csharp_version: str) -> bool:
    latest = "latest"
    run_id = int(time.time() * 1000.0)
    print(
        f"Running training with python={python_version or latest} and c#={csharp_version or latest}"
    )
    output_dir = "models" if python_version else "results"
    nn_file_expected = f"./{output_dir}/{run_id}/3DBall.nn"
    onnx_file_expected = f"./{output_dir}/{run_id}/3DBall.onnx"
    frozen_graph_file_expected = f"./{output_dir}/{run_id}/3DBall/frozen_graph_def.pb"

    if os.path.exists(nn_file_expected):
        # Should never happen - make sure nothing leftover from an old test.
        print("Artifacts from previous build found!")
        return False

    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    # Only build the standalone player if we're overriding the C# version
    # Otherwise we'll use the one built earlier in the pipeline.
    if csharp_version is not None:
        # We can't rely on the old C# code recognizing the commandline argument to set the output
        # So rename testPlayer (containing the most recent build) to something else temporarily
        artifact_path = get_base_output_path()
        full_player_path = os.path.join(artifact_path, "testPlayer.app")
        temp_player_path = os.path.join(artifact_path, "temp_testPlayer.app")
        final_player_path = os.path.join(
            artifact_path, f"testPlayer_{csharp_version}.app"
        )

        os.rename(full_player_path, temp_player_path)

        checkout_csharp_version(csharp_version)
        build_returncode = run_standalone_build(base_path)

        if build_returncode != 0:
            print(f"Standalone build FAILED! with return code {build_returncode}")
            return False

        # Now rename the newly-built executable, and restore the old one
        os.rename(full_player_path, final_player_path)
        os.rename(temp_player_path, full_player_path)
        standalone_player_path = f"testPlayer_{csharp_version}"
    else:
        standalone_player_path = "testPlayer"

    venv_path = init_venv(python_version)

    # Copy the default training config but override the max_steps parameter,
    # and reduce the batch_size and buffer_size enough to ensure an update step happens.
    yaml_out = "override.yaml"
    if python_version:
        overrides: Any = {"max_steps": 100, "batch_size": 10, "buffer_size": 10}
        override_legacy_config_file(
            python_version, "config/trainer_config.yaml", yaml_out, **overrides
        )
    else:
        overrides = {
            "hyperparameters": {"batch_size": 10, "buffer_size": 10},
            "max_steps": 100,
        }
        override_config_file("config/ppo/3DBall.yaml", yaml_out, overrides)

    env_path = os.path.join(get_base_output_path(), standalone_player_path + ".app")
    mla_learn_cmd = (
        f"mlagents-learn {yaml_out} --force --env={env_path} "
        f"--run-id={run_id} --no-graphics --env-args -logFile -"
    )  # noqa
    res = subprocess.run(
        f"source {venv_path}/bin/activate; {mla_learn_cmd}", shell=True
    )

    # Save models as artifacts (only if we're using latest python and C#)
    if csharp_version is None and python_version is None:
        model_artifacts_dir = os.path.join(get_base_output_path(), "models")
        os.makedirs(model_artifacts_dir, exist_ok=True)
        shutil.copy(nn_file_expected, model_artifacts_dir)
        shutil.copy(onnx_file_expected, model_artifacts_dir)
        shutil.copy(frozen_graph_file_expected, model_artifacts_dir)

    if (
        res.returncode != 0
        or not os.path.exists(nn_file_expected)
        or not os.path.exists(onnx_file_expected)
    ):
        print("mlagents-learn run FAILED!")
        return False

    if csharp_version is None and python_version is None:
        # Use abs path so that loading doesn't get confused
        model_path = os.path.abspath(os.path.dirname(nn_file_expected))
        # Onnx loading for overrides not currently supported, but this is
        # where to add it in when it is.
        for extension in ["nn"]:
            inference_ok = run_inference(env_path, model_path, extension)
            if not inference_ok:
                return False

    print("mlagents-learn run SUCCEEDED!")
    return True


def run_inference(env_path: str, output_path: str, model_extension: str) -> bool:
    start_time = time.time()
    exes = find_executables(env_path)
    if len(exes) != 1:
        print(f"Can't determine the player executable in {env_path}. Found {exes}.")
        return False

    log_output_path = f"{get_base_output_path()}/inference.{model_extension}.txt"

    exe_path = exes[0]
    args = [
        exe_path,
        "-nographics",
        "-batchmode",
        "-logfile",
        log_output_path,
        "--mlagents-override-model-directory",
        output_path,
        "--mlagents-quit-on-load-failure",
        "--mlagents-quit-after-episodes",
        "1",
        "--mlagents-override-model-extension",
        model_extension,
    ]
    res = subprocess.run(args)
    end_time = time.time()
    if res.returncode != 0:
        print("Error running inference!")
        print("Command line: " + " ".join(args))
        subprocess.run(["cat", log_output_path])
        return False
    else:
        print(f"Inference succeeded! Took {end_time - start_time} seconds")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None)
    parser.add_argument("--csharp", default=None)
    args = parser.parse_args()

    try:
        ok = run_training(args.python, args.csharp)
        if not ok:
            sys.exit(1)

    finally:
        # Cleanup - this gets executed even if we hit sys.exit()
        undo_git_checkout()


if __name__ == "__main__":
    main()
