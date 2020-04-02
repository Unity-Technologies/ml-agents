import argparse
import os
import sys
import subprocess
import time

from .yamato_utils import (
    get_base_path,
    run_standalone_build,
    init_venv,
    override_config_file,
    checkout_csharp_version,
    undo_git_checkout,
)


def run_training(python_version, csharp_version):
    latest = "latest"
    run_id = int(time.time() * 1000.0)
    print(
        f"Running training with python={python_version or latest} and c#={csharp_version or latest}"
    )
    nn_file_expected = f"./models/{run_id}/3DBall.nn"
    if os.path.exists(nn_file_expected):
        # Should never happen - make sure nothing leftover from an old test.
        print("Artifacts from previous build found!")
        sys.exit(1)

    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    # Only build the standalone player if we're overriding the C# version
    # Otherwise we'll use the one built earlier in the pipeline.
    if csharp_version is not None:
        # We can't rely on the old C# code recognizing the commandline argument to set the output
        # So rename testPlayer (containing the most recent build) to something else temporarily
        full_player_path = os.path.join("Project", "testPlayer.app")
        temp_player_path = os.path.join("Project", "temp_testPlayer.app")
        final_player_path = os.path.join("Project", f"testPlayer_{csharp_version}.app")

        os.rename(full_player_path, temp_player_path)

        checkout_csharp_version(csharp_version)
        build_returncode = run_standalone_build(base_path)

        if build_returncode != 0:
            print("Standalone build FAILED!")
            sys.exit(build_returncode)

        # Now rename the newly-built executable, and restore the old one
        os.rename(full_player_path, final_player_path)
        os.rename(temp_player_path, full_player_path)
        standalone_player_path = f"testPlayer_{csharp_version}"
    else:
        standalone_player_path = "testPlayer"

    venv_path = init_venv(python_version)

    # Copy the default training config but override the max_steps parameter,
    # and reduce the batch_size and buffer_size enough to ensure an update step happens.
    override_config_file(
        "config/trainer_config.yaml",
        "override.yaml",
        max_steps=100,
        batch_size=10,
        buffer_size=10,
    )

    mla_learn_cmd = (
        f"mlagents-learn override.yaml --train --env=Project/{standalone_player_path} "
        f"--run-id={run_id} --no-graphics --env-args -logFile -"
    )  # noqa
    res = subprocess.run(
        f"source {venv_path}/bin/activate; {mla_learn_cmd}", shell=True
    )

    if res.returncode != 0 or not os.path.exists(nn_file_expected):
        print("mlagents-learn run FAILED!")
        sys.exit(1)

    print("mlagents-learn run SUCCEEDED!")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None)
    parser.add_argument("--csharp", default=None)
    args = parser.parse_args()

    try:
        run_training(args.python, args.csharp)
    finally:
        # Cleanup - this gets executed even if we hit sys.exit()
        undo_git_checkout()


if __name__ == "__main__":
    main()
