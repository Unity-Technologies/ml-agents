import argparse

from .yamato_utils import init_venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlagents-version", default=None)
    parser.add_argument("--extra-packages", default=None)
    args = parser.parse_args()
    extra_packages = []
    if args.extra_packages is not None:
        extra_packages = args.extra_packages.split(",")

    init_venv(
        mlagents_python_version=args.mlagents_version, extra_packages=extra_packages
    )


if __name__ == "__main__":
    main()
