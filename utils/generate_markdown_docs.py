#!/usr/bin/env python3
import os
import subprocess
import sys
import yaml
import argparse
import hashlib


# pydoc-markdown -I . -m module_name --render_toc > doc.md


def hash_file(filename):
    """
    Calculate the md5 hash of a file. Used to check for stale files.

    :param filename: The name of the file to check
    :type str:
    :return: A string containing the md5 hash of the file
    :rtype: str
    """
    if os.path.exists(filename):
        hasher = hashlib.md5()
        with open(filename, "rb") as file_to_hash:
            buffer = file_to_hash.read()
            hasher.update(buffer)
        return hasher.hexdigest()
    else:
        return 0


def remove_trailing_whitespace(filename):
    """
    Removes trailing whitespace from a file.

    :param filename: The name of the file to process
    :type str:
    """
    num_changed = 0
    # open the source file
    with open(filename, "rb") as f:
        source_file = f.read().decode()
    # grab all the lines, removing the trailing whitespace
    lines = [line.rstrip() for line in source_file.splitlines()]

    # process lines to construct destination file
    while lines and not lines[-1]:
        lines.pop(-1)
    lines.append("")
    destination_file = "\n".join(lines)

    # compare source and destination and write only if changed
    if source_file != destination_file:
        num_changed += 1
        with open(filename, "wb") as f:
            f.write(destination_file.encode())


if __name__ == "__main__":
    """
    Pre-commit hook to generate Python API documentation using pydoc-markdown
    and write as markdown files. Each package should have a config file,
    pydoc-config.yaml, at the root level that provides configurations for
    the generation. Fails if documentation was updated, passes otherwise. This
    allows for pre-commit to fail in CI/CD and makes sure dev commits the doc
    updates.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dirs", nargs="+")
    args = parser.parse_args()

    ok = True
    for package_dir in args.package_dirs:
        config_path = os.path.join(os.getcwd(), package_dir, "pydoc-config.yaml")
        print(config_path)
        with open(config_path) as config_file:
            config = yaml.full_load(config_file)
            for module in config["modules"]:
                module_name = module["name"]
                submodules = module["submodules"]
                output_file_name = f"./{config['folder']}/{module['file_name']}"
                old_hash = hash_file(output_file_name)
                module_args = []
                for submodule in submodules:
                    module_args.append("-m")
                    module_args.append(f"{module_name}.{submodule}")
                with open(output_file_name, "w") as output_file:
                    subprocess_args = [
                        "pydoc-markdown",
                        "-I",
                        f"./{package_dir}",
                        *module_args,
                        "--render-toc",
                    ]
                    subprocess.check_call(subprocess_args, stdout=output_file)
                remove_trailing_whitespace(output_file_name)
                new_hash = hash_file(output_file_name)
                ok &= old_hash == new_hash

    sys.exit(0 if ok else 1)
