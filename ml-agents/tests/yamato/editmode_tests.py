import os
import sys
import subprocess
import shutil
import xml.dom.minidom
from typing import NamedTuple

from .yamato_utils import get_base_path, get_unity_executable_path


def clean_previous_results(base_path):
    """
    Clean up old results and make the artifacts path.
    """
    artifacts_path = os.path.join(base_path, "artifacts/")
    results_xml_path = os.path.join(base_path, "results.xml")

    if os.path.exists(results_xml_path):
        os.remove(results_xml_path)

    if os.path.exists(artifacts_path):
        os.rmdir(artifacts_path)
    os.mkdir(artifacts_path)


class TestResults(NamedTuple):
    total: str
    passed: str
    failed: str
    duration: str


def parse_results(results_xml):
    """
    Extract the test results from the xml file.
    """
    stats = {}
    dom_tree = xml.dom.minidom.parse(results_xml)
    collection = dom_tree.documentElement
    for attribute in ["total", "passed", "failed", "duration"]:
        stats[attribute] = collection.getAttribute(attribute)
    return TestResults(**stats)


def main():
    base_path = get_base_path()
    artifacts_path = os.path.join(base_path, "artifacts/")
    results_xml_path = os.path.join(base_path, "results.xml")
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
        f"{base_path}/Project",
        "-testResults",
        f"{base_path}/results.xml",
        "-testPlatform",
        "editmode",
    ]
    print(f"{' '.join(test_args)} ...")

    timeout = 30 * 60  # 30 minutes, just in case
    res: subprocess.CompletedProcess = subprocess.run(test_args, timeout=timeout)

    stats = parse_results(results_xml_path)
    print(
        f"{stats.total} tests executed in {stats.duration}s: {stats.passed} passed, "
        f"{stats.failed} failed. More details in results.xml"
    )

    try:
        # copy results to artifacts dir
        shutil.copy2(results_xml_path, artifacts_path)
    except Exception:
        pass

    if res.returncode == 0 and os.path.exists(results_xml_path):
        print("Test run SUCCEEDED!")
    else:
        print("Test run FAILED!")

    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
