#!/usr/bin/env python

import os
import sys
from typing import Dict

VERSION_LINE_START = "VERSION = "

DIRECTORIES = ["ml-agents", "ml-agents-envs", "gym-unity"]


def extract_version_string(filename):
    with open(filename) as f:
        for l in f.readlines():
            if l.startswith(VERSION_LINE_START):
                return l.replace(VERSION_LINE_START, "").strip()
    return None


def check_versions() -> bool:
    version_by_dir: Dict[str, str] = {}
    for directory in DIRECTORIES:
        path = os.path.join(directory, "setup.py")
        version = extract_version_string(path)
        print(f"Found version {version} for {directory}")
        version_by_dir[directory] = version

    # Make sure we have exactly one version, and it's not none
    versions = set(version_by_dir.values())
    if len(versions) != 1 or None in versions:
        print("Each setup.py must have the same VERSION string.")
        return False
    return True


if __name__ == "__main__":
    ok = check_versions()
    return_code = 0 if ok else 1
    sys.exit(return_code)
