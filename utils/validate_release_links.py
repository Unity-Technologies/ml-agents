#!/usr/bin/env python3

import ast
import sys
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Pattern

RELEASE_PATTERN = re.compile(r"release_[0-9]+(_docs)*")
TRAINER_INIT_FILE = "ml-agents/mlagents/trainers/__init__.py"

MATCH_ANY = re.compile(r"(?s).*")
# Filename -> regex list to allow specific lines.
# To allow everything in the file, use None for the value
ALLOW_LIST = {
    # Previous release table
    "README.md": re.compile(r"\*\*(Verified Package ([0-9]\.?)*|Release [0-9]+)\*\*"),
    "docs/Versioning.md": MATCH_ANY,
    "com.unity.ml-agents/CHANGELOG.md": MATCH_ANY,
    "utils/make_readme_table.py": MATCH_ANY,
    "utils/validate_release_links.py": MATCH_ANY,
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
    This will be None on the main branch.
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


def check_file(
    filename: str, global_allow_pattern: Pattern, release_tag: str
) -> List[str]:
    """
    Validate a single file and return any offending lines.
    """
    bad_lines = []
    with tempfile.TemporaryDirectory() as tempdir:
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        new_file_name = os.path.join(tempdir, os.path.basename(filename))
        with open(new_file_name, "w+") as new_file:
            # default to match everything if there is nothing in the ALLOW_LIST
            allow_list_pattern = ALLOW_LIST.get(filename, None)
            with open(filename) as f:
                for line in f:
                    keep_line = True
                    keep_line = not RELEASE_PATTERN.search(line)
                    keep_line |= global_allow_pattern.search(line) is not None
                    keep_line |= (
                        allow_list_pattern is not None
                        and allow_list_pattern.search(line) is not None
                    )

                    if keep_line:
                        new_file.write(line)
                    else:
                        bad_lines.append(f"{filename}: {line}")
                        new_file.write(
                            re.sub(r"release_[0-9]+", fr"{release_tag}", line)
                        )
        if bad_lines:
            if os.path.exists(filename):
                os.remove(filename)
                os.rename(new_file_name, filename)

    return bad_lines


def check_all_files(allow_pattern: Pattern, release_tag: str) -> List[str]:
    """
    Validate all files tracked by git.
    :param allow_pattern:
    """
    bad_lines = []
    file_types = {".py", ".md", ".cs"}
    for file_name in git_ls_files():
        if "localized" in file_name or os.path.splitext(file_name)[1] not in file_types:
            continue
        bad_lines += check_file(file_name, allow_pattern, release_tag)
    return bad_lines


def main():
    release_tag = get_release_tag()
    if not release_tag:
        print("Release tag is None, exiting")
        sys.exit(0)

    print(f"Release tag: {release_tag}")
    allow_pattern = re.compile(f"{release_tag}(_docs)*")
    bad_lines = check_all_files(allow_pattern, release_tag)
    if bad_lines:
        for line in bad_lines:
            print(line)

        print("*************************************************************")
        print(
            "This script attempted to fix the above errors. Please double "
            + "check them to make sure the replacements were done correctly"
        )

    sys.exit(1 if bad_lines else 0)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_pattern()
    main()
