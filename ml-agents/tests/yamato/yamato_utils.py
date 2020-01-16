import os
import sys


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
