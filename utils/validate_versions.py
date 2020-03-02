#!/usr/bin/env python3

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

UNITY_PACKAGE_JSON_PATH = "com.unity.ml-agents/package.json"
ACADEMY_PATH = "com.unity.ml-agents/Runtime/Academy.cs"


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
    # Package version is a bit stricter - only set it if we're not a "dev" version.
    if "dev" not in new_version:
        package_version = new_version + "-preview"
        print(
            f"Setting package version to {package_version} in {UNITY_PACKAGE_JSON_PATH}"
        )
        set_package_version(package_version)
        print(f"Setting package version to {package_version} in {ACADEMY_PATH}")
        set_academy_version_string(package_version)


def set_package_version(new_version: str) -> None:
    with open(UNITY_PACKAGE_JSON_PATH, "r") as f:
        package_json = json.load(f)
    if "version" in package_json:
        package_json["version"] = new_version
    with open(UNITY_PACKAGE_JSON_PATH, "w") as f:
        json.dump(package_json, f, indent=2)


def set_academy_version_string(new_version):
    needle = "internal const string k_PackageVersion"
    found = 0
    with open(ACADEMY_PATH) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if needle in l:
            left, right = l.split(" = ")
            right = f' = "{new_version}";\n'
            lines[i] = left + right
            found += 1
    if found != 1:
        raise RuntimeError(
            f'Expected to find search string "{needle}" exactly once, but found it {found} times'
        )
    with open(ACADEMY_PATH, "w") as f:
        f.writelines(lines)


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
