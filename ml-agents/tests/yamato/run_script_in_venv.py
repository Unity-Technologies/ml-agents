import sys
import subprocess

from .yamato_utils import get_base_path, init_venv


def run_script_in_venv(script_path):
    """
    Sets up the venv, runs the specified script, and returns based on whether the script succeeded.
    :return:
    """
    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    venv_path = init_venv()

    res = subprocess.run(
        f"source {venv_path}/bin/activate; python {script_path}", shell=True
    )

    if res.returncode != 0:
        print(f"{script_path} run FAILED!")
        sys.exit(1)

    print("{script_path} run SUCCEEDED!")
    sys.exit(0)


def main():
    script_path = sys.argv[1]
    run_script_in_venv(script_path)


if __name__ == "__main__":
    main()
