#!/usr/bin/env python3

import os
import glob
from setuptools import find_packages, PEP420PackageFinder


class NonTrivialPEP420PackageFinder(PEP420PackageFinder):
    """
    The PEP420PackageFinder (used by find_namespace_packages) thinks everything
    looks like a package, even if there are no python files in it. This is a
    little stricter and only considers directories with python files in it.
    """

    @staticmethod
    def _looks_like_package(path):
        glob_path = os.path.join(path, "*.py")
        return any(glob.iglob(glob_path))


def validate_packages(root_dir):
    """
    Makes sure that all python files are discoverable by find_packages(), which
    is what we use in setup.py. We could potentially use
    find_namespace_packages instead, but depending on PEP420 has been flaky
    in the past (particularly with regards to mypy).
    """
    exclude = ["*.tests", "*.tests.*", "tests.*", "tests"]
    found_packages = find_packages(root_dir, exclude=exclude)
    found_ns_packages = NonTrivialPEP420PackageFinder.find(root_dir, exclude=exclude)
    assert found_packages, f"Couldn't find anything in directory {root_dir}"
    if set(found_packages) != set(found_ns_packages):
        raise RuntimeError(
            "The following packages are not discoverable using found_packages():\n"
            f"{set(found_ns_packages) - set(found_packages)}\n"
            "Make sure you have an __init__.py file in the directories."
        )
    else:
        print(f"__init__.py files for {root_dir} are OK.")


def main():
    for root_dir in ["ml-agents", "ml-agents-envs", "gym-unity"]:
        validate_packages(root_dir)


if __name__ == "__main__":
    main()
