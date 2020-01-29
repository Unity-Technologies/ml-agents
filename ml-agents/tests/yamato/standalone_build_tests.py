import sys
import subprocess

from .yamato_utils import get_base_path, get_unity_executable_path


def main():
    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    unity_exe = get_unity_executable_path()
    print(f"Starting tests via {unity_exe}")

    test_args = [
        unity_exe,
        "-projectPath",
        f"{base_path}/Project",
        "-logfile",
        "-",
        "-batchmode",
        "-executeMethod",
        "MLAgents.StandaloneBuildTest.BuildStandalonePlayerOSX",
    ]
    print(f"{' '.join(test_args)} ...")

    timeout = 30 * 60  # 30 minutes, just in case
    res: subprocess.CompletedProcess = subprocess.run(test_args, timeout=timeout)

    if res.returncode == 0:
        print("Test run SUCCEEDED!")
    else:
        print("Test run FAILED!")

    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
