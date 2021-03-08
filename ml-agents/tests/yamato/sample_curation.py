import sys
import argparse

from .yamato_utils import get_base_path, create_samples


def main(scenes):
    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    returncode = create_samples(
        scenes,
        base_path,
        log_output_path=None,  # Log to stdout so we get timestamps on the logs
    )

    if returncode == 0:
        print("Test run SUCCEEDED!")
    else:
        print("Test run FAILED!")

    sys.exit(returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", nargs="+", default=None, required=True)
    args = parser.parse_args()
    main(args.scene)
