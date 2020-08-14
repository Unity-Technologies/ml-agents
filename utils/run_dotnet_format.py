#!/usr/bin/env python3

import argparse
import os
import subprocess
import tempfile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tempdir:
        # Could potentially hit the commandline limit, so write files to a response file
        # See https://github.com/dotnet/format/issues/699
        resp_file = os.path.join(tempdir, "response.txt")
        with open(resp_file, "w") as fp:
            for f in args.files:
                fp.write(f + "\n")

        subprocess_args = ["dotnet", "format", "--folder", "--include", f"@{resp_file}"]
        subprocess.check_call(subprocess_args)
