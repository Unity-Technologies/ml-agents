import os
import subprocess
import yaml


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


def run_standalone_build(base_path: str, verbose: bool = False) -> int:
    """
    Run BuildStandalonePlayerOSX test to produce a player at Project/testPlayer
    :param base_path:
    :return:
    """
    unity_exe = get_unity_executable_path()
    print(f"Running BuildStandalonePlayerOSX via {unity_exe}")

    test_args = [
        unity_exe,
        "-projectPath",
        f"{base_path}/Project",
        "-batchmode",
        "-executeMethod",
        "MLAgents.StandaloneBuildTest.BuildStandalonePlayerOSX",
    ]
    if verbose:
        test_args += ["-logfile", "-"]
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
            f"source venv/bin/activate; python -m pip install -q {cmd}", shell=True
        )


def override_config_file(src_path, dest_path, **kwargs):
    """
    Override settings in a trainer config file. For example,
        override_config_file(src_path, dest_path, max_steps=42)
    will copy the config file at src_path to dest_path, but override the max_steps field to 42 for all brains.
    """
    with open(src_path) as f:
        configs = yaml.safe_load(f)

    for config in configs.values():
        config.update(**kwargs)

    with open(dest_path, "w") as f:
        yaml.dump(configs, f)
