#!/usr/bin/env python3

import ast
import sys
import os
import re
import subprocess
from typing import List, Optional, Pattern

RELEASE_PATTERN = re.compile(r"release_[0-9]+(_docs)*")
TRAINER_INIT_FILE = "ml-agents/mlagents/trainers/__init__.py"

# Filename -> regex list to allow specific lines.
# To allow everything in the file, use None for the value
ALLOW_LIST = {
    # Previous release table
    "README.md": re.compile(r"\*\*Release [0-9]+\*\*"),
    "docs/Versioning.md": None,
    "com.unity.ml-agents/CHANGELOG.md": None,
    "utils/make_readme_table.py": None,
    "utils/validate_release_links.py": None,
}


def test_pattern():
    # Just some sanity check that the regex works as expected.
    assert RELEASE_PATTERN.search(
        "https://github.com/Unity-Technologies/ml-agents/blob/release_4_docs/Food.md"
    )
    assert RELEASE_PATTERN.search(
        "https://github.com/Unity-Technologies/ml-agents/blob/release_4/Foo.md"
    )
    assert RELEASE_PATTERN.search(
        "git clone --branch release_4 https://github.com/Unity-Technologies/ml-agents.git"
    )
    assert RELEASE_PATTERN.search(
        "https://github.com/Unity-Technologies/ml-agents/blob/release_123_docs/Foo.md"
    )
    assert RELEASE_PATTERN.search(
        "https://github.com/Unity-Technologies/ml-agents/blob/release_123/Foo.md"
    )
    assert not RELEASE_PATTERN.search(
        "https://github.com/Unity-Technologies/ml-agents/blob/latest_release/docs/Foo.md"
    )
    print("tests OK!")


def git_ls_files() -> List[str]:
    """
    Run "git ls-files" and return a list with one entry per line.
    This returns the list of all files tracked by git.
    """
    return subprocess.check_output(["git", "ls-files"], universal_newlines=True).split(
        "\n"
    )


def get_release_tag() -> Optional[str]:
    """
    Returns the release tag for the mlagents python package.
    This will be None on the master branch.
    :return:
    """
    with open(TRAINER_INIT_FILE) as f:
        for line in f:
            if "__release_tag__" in line:
                lhs, equals_string, rhs = line.strip().partition(" = ")
                # Evaluate the right hand side of the expression
                return ast.literal_eval(rhs)
    # If we couldn't find the release tag, raise an exception
    # (since we can't return None here)
    raise RuntimeError("Can't determine release tag")


def check_file(filename: str, global_allow_pattern: Pattern) -> List[str]:
    """
    Validate a single file and return any offending lines.
    """
    bad_lines = []
    with open(filename) as f:
        for line in f:
            if not RELEASE_PATTERN.search(line):
                continue

            if global_allow_pattern.search(line):
                continue

            if filename in ALLOW_LIST:
                if ALLOW_LIST[filename] is None or ALLOW_LIST[filename].search(line):
                    continue

            bad_lines.append(f"{filename}: {line.strip()}")
    return bad_lines


def check_all_files(allow_pattern: Pattern) -> List[str]:
    """
    Validate all files tracked by git.
    :param allow_pattern:
    """
    bad_lines = []
    file_types = {".py", ".md", ".cs"}
    for file_name in git_ls_files():
        if "localized" in file_name or os.path.splitext(file_name)[1] not in file_types:
            continue
        bad_lines += check_file(file_name, allow_pattern)
    return bad_lines


def main():
    release_tag = get_release_tag()
    if not release_tag:
        print("Release tag is None, exiting")
        sys.exit(0)

    print(f"Release tag: {release_tag}")
    allow_pattern = re.compile(f"{release_tag}(_docs)*")
    bad_lines = check_all_files(allow_pattern)
    if bad_lines:
        print(
            f"Found lines referring to previous release. Either update the files, or add an exclusion to {__file__}"
        )
        for line in bad_lines:
            print(line)

    sys.exit(1 if bad_lines else 0)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_pattern()
    main()
