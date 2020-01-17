import sys
import subprocess

from .yamato_utils import get_base_path, run_standalone_build, init_venv


def main():
    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    build_returncode = run_standalone_build(base_path)
    if build_returncode != 0:
        print("Standalong build FAILED!")
        sys.exit(build_returncode)

    init_venv()

    # TODO pass exe name to build
    mla_learn_cmd = "mlagents-learn config/trainer_config.yaml --train --env=testPlayer --no-graphics --env-args -logFile -"  # noqa
    res = subprocess.run(f"source venv/bin/activate; {mla_learn_cmd}", shell=True)

    if res.returncode == 0:
        print("mlagents-learn run SUCCEEDED!")
    else:
        print("mlagents-learn run FAILED!")

    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
