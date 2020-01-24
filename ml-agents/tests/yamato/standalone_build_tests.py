import sys

from .yamato_utils import get_base_path, run_standalone_build


def main():
    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    returncode = run_standalone_build(base_path, verbose=True)

    if returncode == 0:
        print("Test run SUCCEEDED!")
    else:
        print("Test run FAILED!")

    sys.exit(returncode)


if __name__ == "__main__":
    main()
