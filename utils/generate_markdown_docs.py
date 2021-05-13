#!/usr/bin/env python3
import os
import subprocess
import sys
import yaml
import argparse
import hashlib


# pydoc-markdown -I . -m module_name --render_toc > doc.md


def hash_file(filename):
    if os.path.exists(filename):
        hasher = hashlib.md5()
        with open(filename, "rb") as file_to_hash:
            buffer = file_to_hash.read()
            hasher.update(buffer)
        return hasher.hexdigest()
    else:
        return 0


def remove_trailing_whitespace(filename):
    nchanged = 0
    with open(filename, "rb") as f:
        code1 = f.read().decode()
    lines = [line.rstrip() for line in code1.splitlines()]
    while lines and not lines[-1]:
        lines.pop(-1)
    lines.append("")  # always end with a newline
    code2 = "\n".join(lines)
    if code1 != code2:
        nchanged += 1
        with open(filename, "wb") as f:
            f.write(code2.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dirs", nargs="+")
    args = parser.parse_args()

    ok = False
    return_code = 0
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
                ok = old_hash == new_hash

        return_code = 0 if ok else 1

    sys.exit(return_code)
