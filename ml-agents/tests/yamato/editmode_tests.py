import os
import sys
import subprocess
from typing import NamedTuple


def get_unity_executable_path():
    UNITY_VERSION = os.environ["UNITY_VERSION"]
    BOKKEN_UNITY = f"/Users/bokken/{UNITY_VERSION}/Unity.app/Contents/MacOS/Unity"
    HUB_UNITY = (
        f"/Applications/Unity/Hub/Editor/{UNITY_VERSION}/Unity.app/Contents/MacOS/Unity"
    )
    if os.path.exists(BOKKEN_UNITY):
        return BOKKEN_UNITY
    if os.path.exists(HUB_UNITY):
        return HUB_UNITY
    raise FileNotFoundError("Can't find bokken or hub executables")


def get_base_path():
    # E.g. 'ml-agents/tests/yamato/editmode_tests.py'
    script_name = sys.argv[0]

    # E.g. '/Users/yourname/code/ml-agents/ml-agents/tests/yamato/editmode_tests.py'
    abs_path = os.path.abspath(script_name)

    # Remove the script name to get the absolute path of where it's being run from.
    return abs_path.replace("/" + script_name, "")


def clean_previous_results(base_path):
    results = os.path.join(base_path, "results.xml")
    if os.path.exists(results):
        os.remove(results)


class TestResults(NamedTuple):
    total: str
    passed: str
    failed: str
    duration: str


def parse_results():
    # copied-and-pasted
    # TODO parse for real?
    stats = {}

    for stat in ["total", "passed", "failed", "duration"]:
        stats[stat] = subprocess.run(
            f"echo 'cat /test-run/test-suite/@{stat}' | xmllint --shell results.xml | awk -F'[=\"]' '!/>/{{print $(NF-1)}}'",  # noqa
            shell=True,
        ).stdout

    return TestResults(**stats)


def main():
    base_path = get_base_path()
    print(f"Running in base path {base_path}")

    print("Cleaning previous results")
    clean_previous_results(base_path)

    unity_exe = get_unity_executable_path()
    print(f"Starting tests via {unity_exe}")

    test_args = [
        unity_exe,
        "-batchmode",
        "-runTests",
        "-logfile",
        "-",
        "-projectPath",
        f"{base_path}/UnitySDK",
        "-testResults",
        f"{base_path}/results.xml",
        "-testPlatform",
        "editmode",
    ]
    print(f"cmd line = {' '.join(test_args)}")

    timeout = 30 * 60  # 30 minutes
    res: subprocess.CompletedProcess = subprocess.run(test_args, timeout=timeout)

    stats = parse_results()
    print(
        f"{stats.total} tests executed in {stats.duration}s: {stats.passed} passed, "
        f"{stats.failed} failed. More details in results.xml"
    )

    if res.returncode == 0 and os.path.exists(os.path.join(base_path, "results.xml")):
        print("Test run SUCCEEDED!")
        os.remove(os.path.join(base_path, "results.xml"))
    else:
        print("Test run FAILED!")

    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
