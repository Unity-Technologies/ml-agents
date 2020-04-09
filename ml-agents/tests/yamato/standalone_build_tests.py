import sys
import argparse

from .yamato_utils import get_base_path, run_standalone_build


def main(scene_path):
    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    executable_name = None
    if scene_path is not None:
        executable_name = scene_path.strip(".unity")
        executable_name = executable_name.split("/")[-1]
        executable_name = "testPlayer-" + executable_name

    returncode = run_standalone_build(
        base_path, output_path=executable_name, scene_path=scene_path
    )

    if returncode == 0:
        print("Test run SUCCEEDED!")
    else:
        print("Test run FAILED!")

    sys.exit(returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()
    main(args.scene)
