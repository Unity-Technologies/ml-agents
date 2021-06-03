#!/usr/bin/env python3

import ast
import sys
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Pattern

RELEASE_PATTERN = re.compile(r"release_[0-9]+(_docs)*")
# This matches the various ways to invoke pip: "pip", "pip3", "python -m pip"
# It matches "mlagents" and "mlagents_envs", accessible as group "package"
# and optionally matches the version, e.g. "==1.2.3"
PIP_INSTALL_PATTERN = re.compile(
    r"(python -m )?pip3* install (?P<quiet>-q )?(?P<package>mlagents(_envs)?)(==[0-9]+\.[0-9]+\.[0-9]+(\.dev[0-9]+)?)?"
)
TRAINER_INIT_FILE = "ml-agents/mlagents/trainers/__init__.py"

MATCH_ANY = re.compile(r"(?s).*")
# Filename -> regex list to allow specific lines.
# To allow everything in the file (effectively skipping it), use MATCH_ANY for the value
ALLOW_LIST = {
    # Previous release table
    "README.md": re.compile(r"\*\*(Verified Package ([0-9]\.?)*|Release [0-9]+)\*\*"),
    "docs/Versioning.md": MATCH_ANY,
    "com.unity.ml-agents/CHANGELOG.md": MATCH_ANY,
    "utils/make_readme_table.py": MATCH_ANY,
    "utils/validate_release_links.py": MATCH_ANY,
}


def test_release_pattern():
    # Just some sanity check that the regex works as expected.
    for s, expected in [
        (
            "https://github.com/Unity-Technologies/ml-agents/blob/release_4_docs/Food.md",
            True,
        ),
        ("https://github.com/Unity-Technologies/ml-agents/blob/release_4/Foo.md", True),
        (
            "git clone --branch release_4 https://github.com/Unity-Technologies/ml-agents.git",
            True,
        ),
        (
            "https://github.com/Unity-Technologies/ml-agents/blob/release_123_docs/Foo.md",
            True,
        ),
        (
            "https://github.com/Unity-Technologies/ml-agents/blob/release_123/Foo.md",
            True,
        ),
        (
            "https://github.com/Unity-Technologies/ml-agents/blob/latest_release/docs/Foo.md",
            False,
        ),
    ]:
        assert bool(RELEASE_PATTERN.search(s)) is expected

    print("release tests OK!")


def test_pip_pattern():
    # Just some sanity check that the regex works as expected.
    for s, expected in [
        ("pip install mlagents", True),
        ("pip3 install -q mlagents", True),
        ("python -m pip install mlagents", True),
        ("python -m pip install mlagents==1.2.3", True),
        ("python -m pip install mlagents_envs==1.2.3", True),
        ("python -m pip install mlagents==11.222.3333", True),
        ("python -m pip install mlagents_envs==11.222.3333", True),
    ]:
        assert bool(PIP_INSTALL_PATTERN.search(s)) is expected

    sub_expected = "Try running rm -rf / to install"
    assert sub_expected == PIP_INSTALL_PATTERN.sub(
        "rm -rf /", "Try running python -m pip install mlagents==1.2.3 to install"
    )

    print("pip tests OK!")


def update_pip_install_line(line, package_verion):
    match = PIP_INSTALL_PATTERN.search(line)
    if match is not None:  # if there is a pip install line
        package_name = match.group("package")
        quiet_option = match.group("quiet") or ""
        replacement_version = (
            f"python -m pip install {quiet_option}{package_name}=={package_verion}"
        )
        updated = PIP_INSTALL_PATTERN.sub(replacement_version, line)
        return updated
    else:  # Don't do anything
        return line


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


def get_python_package_version() -> str:
    """
    Returns the mlagents python package.
    :return:
    """
    with open(TRAINER_INIT_FILE) as f:
        for line in f:
            if "__version__" in line:
                lhs, equals_string, rhs = line.strip().partition(" = ")
                # Evaluate the right hand side of the expression
                return ast.literal_eval(rhs)
    # If we couldn't find the release tag, raise an exception
    # (since we can't return None here)
    raise RuntimeError("Can't determine python package version")


def check_file(
    filename: str,
    release_tag_pattern: Pattern,
    release_tag: str,
    pip_allow_pattern: Pattern,
    package_version: str,
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
                    # Does it contain anything of the form release_123
                    has_release_pattern = RELEASE_PATTERN.search(line) is not None
                    # Does it contain this particular release, e.g. release_42 or release_42_docs
                    has_release_tag_pattern = (
                        release_tag_pattern.search(line) is not None
                    )
                    # Does it contain the allow list pattern for the file (if there is one)
                    has_allow_list_pattern = (
                        allow_list_pattern
                        and allow_list_pattern.search(line) is not None
                    )

                    pip_install_ok = (
                        has_allow_list_pattern
                        or PIP_INSTALL_PATTERN.search(line) is None
                        or pip_allow_pattern.search(line) is not None
                    )

                    release_tag_ok = (
                        not has_release_pattern
                        or has_release_tag_pattern
                        or has_allow_list_pattern
                    )

                    if release_tag_ok and pip_install_ok:
                        new_file.write(line)
                    else:
                        bad_lines.append(f"{filename}: {line}")
                        new_line = re.sub(r"release_[0-9]+", fr"{release_tag}", line)
                        new_line = update_pip_install_line(new_line, package_version)
                        new_file.write(new_line)
        if bad_lines:
            if os.path.exists(filename):
                os.remove(filename)
                os.rename(new_file_name, filename)

    return bad_lines


def check_all_files(
    release_allow_pattern: Pattern,
    release_tag: str,
    pip_allow_pattern: Pattern,
    package_version: str,
) -> List[str]:
    """
    Validate all files tracked by git.
    :param release_allow_pattern:
    """
    bad_lines = []
    file_types = {".py", ".md", ".cs", ".ipynb"}
    for file_name in git_ls_files():
        if "localized" in file_name or os.path.splitext(file_name)[1] not in file_types:
            continue
        bad_lines += check_file(
            file_name,
            release_allow_pattern,
            release_tag,
            pip_allow_pattern,
            package_version,
        )
    return bad_lines


def main():
    release_tag = get_release_tag()
    if not release_tag:
        print("Release tag is None, exiting")
        sys.exit(0)

    package_version = get_python_package_version()
    print(f"Release tag: {release_tag}")
    print(f"Python package version: {package_version}")
    release_allow_pattern = re.compile(f"{release_tag}(_docs)?")
    pip_allow_pattern = re.compile(
        fr"python -m pip install (-q )?mlagents(_envs)?=={package_version}"
    )
    bad_lines = check_all_files(
        release_allow_pattern, release_tag, pip_allow_pattern, package_version
    )
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
        test_release_pattern()
        test_pip_pattern()
    main()
