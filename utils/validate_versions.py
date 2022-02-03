#!/usr/bin/env python3

import os
import json
import sys
from typing import Dict, Optional
import argparse

VERSION_LINE_START = "__version__ = "

DIRECTORIES = ["ml-agents/mlagents/trainers", "ml-agents-envs/mlagents_envs"]

MLAGENTS_PACKAGE_JSON_PATH = "com.unity.ml-agents/package.json"
MLAGENTS_EXTENSIONS_PACKAGE_JSON_PATH = "com.unity.ml-agents.extensions/package.json"

ACADEMY_PATH = "com.unity.ml-agents/Runtime/Academy.cs"

PYTHON_VERSION_FILE_TEMPLATE = """# Version of the library that will be used to upload to pypi
__version__ = {version}

# Git tag that will be checked to determine whether to trigger upload to pypi
__release_tag__ = {release_tag}
"""


def _escape_non_none(s: Optional[str]) -> str:
    """
    Returns s escaped in quotes if it is non-None, else "None"
    :param s:
    :return:
    """
    if s is not None:
        return f'"{s}"'
    else:
        return "None"


def extract_version_string(filename):
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith(VERSION_LINE_START):
                return line.replace(VERSION_LINE_START, "").strip()
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


def set_version(
    python_version: str,
    csharp_version: str,
    csharp_extensions_version: str,
    release_tag: Optional[str],
) -> None:
    # Sanity check - make sure test tags have a test or dev version
    if release_tag and "test" in release_tag:
        if not ("dev" in python_version or "test" in python_version):
            raise RuntimeError('Test tags must use a "test" or "dev" version.')

    new_contents = PYTHON_VERSION_FILE_TEMPLATE.format(
        version=_escape_non_none(python_version),
        release_tag=_escape_non_none(release_tag),
    )
    for directory in DIRECTORIES:
        path = os.path.join(directory, "__init__.py")
        print(f"Setting {path} to version {python_version}")
        with open(path, "w") as f:
            f.write(new_contents)

    if csharp_version is not None:
        package_version = f"{csharp_version}-exp.1"
        if csharp_extensions_version is not None:
            # since this has never been promoted we need to keep
            # it in preview forever or CI will fail
            extension_version = f"{csharp_extensions_version}-preview"
        print(
            f"Setting package version to {package_version} in {MLAGENTS_PACKAGE_JSON_PATH}"
            f" and {MLAGENTS_EXTENSIONS_PACKAGE_JSON_PATH}"
        )
        set_package_version(package_version)
        set_extension_package_version(package_version, extension_version)
        print(f"Setting package version to {package_version} in {ACADEMY_PATH}")
        set_academy_version_string(package_version)


def set_package_version(new_version: str) -> None:
    with open(MLAGENTS_PACKAGE_JSON_PATH) as f:
        package_json = json.load(f)
    if "version" in package_json:
        package_json["version"] = new_version
    with open(MLAGENTS_PACKAGE_JSON_PATH, "w") as f:
        json.dump(package_json, f, indent=2)
        f.write("\n")


def set_extension_package_version(
    new_dependency_version: str, new_extension_version
) -> None:
    with open(MLAGENTS_EXTENSIONS_PACKAGE_JSON_PATH) as f:
        package_json = json.load(f)
    package_json["dependencies"]["com.unity.ml-agents"] = new_dependency_version
    if new_extension_version is not None:
        package_json["version"] = new_extension_version
    with open(MLAGENTS_EXTENSIONS_PACKAGE_JSON_PATH, "w") as f:
        json.dump(package_json, f, indent=2)
        f.write("\n")


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


def print_release_tag_commands(
    python_version: str, csharp_version: str, release_tag: str
):
    python_tag = f"python-packages_{python_version}"
    csharp_tag = f"com.unity.ml-agents_{csharp_version}"
    docs_tag = f"{release_tag}_docs"
    print(
        f"""
###
Use these commands to create the tags after the release:
###
git checkout {release_tag}
git tag -f latest_release
git push -f origin latest_release
git tag -f {docs_tag}
git push -f origin {docs_tag}
git tag {python_tag}
git push -f origin {python_tag}
git tag {csharp_tag}
git push -f origin {csharp_tag}
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-version", default=None)
    parser.add_argument("--csharp-version", default=None)
    parser.add_argument("--csharp-extensions-version", default=None)
    parser.add_argument("--release-tag", default=None)
    # unused, but allows precommit to pass filenames
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    if args.python_version:
        print(f"Updating python library to version {args.python_version}")
        if args.csharp_version:
            print(f"Updating C# package to version {args.csharp_version}")
        if args.csharp_extensions_version:
            print(
                f"Updating C# extensions package to version {args.csharp_extensions_version}"
            )
        set_version(
            args.python_version,
            args.csharp_version,
            args.csharp_extensions_version,
            args.release_tag,
        )
        if args.release_tag is not None:
            print_release_tag_commands(
                args.python_version, args.csharp_version, args.release_tag
            )
    else:
        ok = check_versions()
        return_code = 0 if ok else 1
        sys.exit(return_code)
