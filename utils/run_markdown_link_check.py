#!/usr/bin/env python3

import argparse
import subprocess

if __name__ == "__main__":
    # markdown-link-check doesn't support multiple files on the commandline, so this hacks around that.
    # Note that you must install the package separately via npm. For example:
    #  brew install npm; npm install -g markdown-link-check
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-remote", action="store_true")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    config_file = (
        "markdown-link-check.full.json"
        if args.check_remote
        else "markdown-link-check.fast.json"
    )

    for f in args.files:
        subprocess_args = ["markdown-link-check", "-c", config_file, f]
        subprocess.check_call(subprocess_args)
