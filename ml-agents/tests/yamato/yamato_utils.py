import os
import subprocess


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
    # We might need to do some more work here if the working directory ever changes
    # E.g. take the full path and back out the main module main.
    # But for now, this should work
    return os.getcwd()


def run_standalone_build(base_path: str) -> int:
    unity_exe = get_unity_executable_path()
    print(f"Running BuildStandalonePlayerOSX via {unity_exe}")

    test_args = [
        unity_exe,
        "-projectPath",
        f"{base_path}/UnitySDK",
        "-logfile",
        "-",
        "-batchmode",
        "-executeMethod",
        "MLAgents.StandaloneBuildTest.BuildStandalonePlayerOSX",
    ]
    print(f"{' '.join(test_args)} ...")

    timeout = 30 * 60  # 30 minutes, just in case
    res: subprocess.CompletedProcess = subprocess.run(test_args, timeout=timeout)
    return res.returncode


def init_venv():
    # Set up the venv and install mlagents
    subprocess.check_call("python -m venv venv", shell=True)
    pip_commands = [
        "--upgrade pip",
        "--upgrade setuptools",
        # TODO build these and publish to internal pypi
        "~/tensorflow_pkg/tensorflow-2.0.0-cp37-cp37m-macosx_10_14_x86_64.whl",
        "-e ./ml-agents-envs",
        "-e ./ml-agents",
    ]
    for cmd in pip_commands:
        subprocess.check_call(
            f"source venv/bin/activate; python -m pip install {cmd}", shell=True
        )
