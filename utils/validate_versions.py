#!/usr/bin/env python

import os
import json
import sys
from typing import Dict
import argparse

VERSION_LINE_START = "__version__ = "

DIRECTORIES = [
    "ml-agents/mlagents/trainers",
    "ml-agents-envs/mlagents_envs",
    "gym-unity/gym_unity",
]

UNITY_PACKAGE_JSON = "com.unity.ml-agents/package.json"


def extract_version_string(filename):
    with open(filename) as f:
        for l in f.readlines():
            if l.startswith(VERSION_LINE_START):
                return l.replace(VERSION_LINE_START, "").strip()
    return None


def check_versions() -> bool:
    version_by_dir: Dict[str, str] = {}
    for directory in DIRECTORIES:
        path = os.path.join(directory, "__init__.py")
        version = extract_version_string(path)
        print(f"Found version {version} for {directory}")
        version_by_dir[directory] = version

    # Make sure we have exactly one version, and it's not none
    versions = set(version_by_dir.values())
    if len(versions) != 1 or None in versions:
        print("Each setup.py must have the same VERSION string.")
        return False
    return True


def set_version(new_version: str) -> None:
    new_contents = f'{VERSION_LINE_START}"{new_version}"\n'
    for directory in DIRECTORIES:
        path = os.path.join(directory, "__init__.py")
        print(f"Setting {path} to version {new_version}")
        with open(path, "w") as f:
            f.write(new_contents)


def set_package_version(new_version: str) -> None:
    with open(UNITY_PACKAGE_JSON, "r") as f:
        package_json = json.load(f)
    if "version" in package_json:
        package_json["version"] = new_version
    out_temp = "new_" + UNITY_PACKAGE_JSON
    with open(out_temp, "w") as f:
        json.dump(package_json, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-version", default=None)
    # unused, but allows precommit to pass filenames
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    if args.new_version:
        print(f"Updating to verison {args.new_version}")
        set_version(args.new_version)
    else:
        ok = check_versions()
        return_code = 0 if ok else 1
        sys.exit(return_code)
