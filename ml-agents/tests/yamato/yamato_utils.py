import os
import shutil
import subprocess
import yaml
from typing import List, Optional


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


def get_base_output_path():
    """"
    Returns the artifact folder to use for yamato jobs.
    """
    return os.path.join(get_base_path(), "artifacts")


def run_standalone_build(
    base_path: str,
    verbose: bool = False,
    output_path: str = None,
    scene_path: str = None,
    log_output_path: str = f"{get_base_output_path()}/standalone_build.txt",
) -> int:
    """
    Run BuildStandalonePlayerOSX test to produce a player. The location defaults to
    artifacts/standalone_build/testPlayer.
    """
    unity_exe = get_unity_executable_path()
    print(f"Running BuildStandalonePlayerOSX via {unity_exe}")

    test_args = [
        unity_exe,
        "-projectPath",
        f"{base_path}/Project",
        "-batchmode",
        "-executeMethod",
        "Unity.MLAgents.StandaloneBuildTest.BuildStandalonePlayerOSX",
    ]

    os.makedirs(os.path.dirname(log_output_path), exist_ok=True)
    subprocess.run(["touch", log_output_path])
    test_args += ["-logfile", log_output_path]

    if output_path is not None:
        output_path = os.path.join(get_base_output_path(), output_path)
        test_args += ["--mlagents-build-output-path", output_path]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if scene_path is not None:
        test_args += ["--mlagents-build-scene-path", scene_path]
    print(f"{' '.join(test_args)} ...")

    timeout = 30 * 60  # 30 minutes, just in case
    res: subprocess.CompletedProcess = subprocess.run(test_args, timeout=timeout)

    # Copy the default build name into the artifacts folder.
    if output_path is None and res.returncode == 0:
        shutil.move(
            os.path.join(base_path, "Project", "testPlayer.app"),
            os.path.join(get_base_output_path(), "testPlayer.app"),
        )

    # Print if we fail or want verbosity.
    if verbose or res.returncode != 0:
        subprocess.run(["cat", log_output_path])

    return res.returncode


def init_venv(
    mlagents_python_version: str = None, extra_packages: Optional[List[str]] = None
) -> str:
    """
    Set up the virtual environment, and return the venv path.
    :param mlagents_python_version: The version of mlagents python packcage to install.
        If None, will do a local install, otherwise will install from pypi
    :return:
    """
    # Use a different venv path for different versions
    venv_path = "venv"
    if mlagents_python_version:
        venv_path += "_" + mlagents_python_version

    # Set up the venv and install mlagents
    subprocess.check_call(f"python -m venv {venv_path}", shell=True)
    pip_commands = [
        "--upgrade pip",
        "--upgrade setuptools",
        # TODO build these and publish to internal pypi
        "~/tensorflow_pkg/tensorflow-2.0.0-cp37-cp37m-macosx_10_14_x86_64.whl",
    ]
    if mlagents_python_version:
        # install from pypi
        pip_commands += [
            f"mlagents=={mlagents_python_version}",
            f"gym-unity=={mlagents_python_version}",
        ]
    else:
        # Local install
        pip_commands += ["-e ./ml-agents-envs", "-e ./ml-agents", "-e ./gym-unity"]
    if extra_packages:
        pip_commands += extra_packages
    for cmd in pip_commands:
        subprocess.check_call(
            f"source {venv_path}/bin/activate; python -m pip install -q {cmd}",
            shell=True,
        )
    return venv_path


def checkout_csharp_version(csharp_version):
    """
    Checks out the specific git revision (usually a tag) for the C# package and Project.
    If csharp_version is None, no changes are made.
    :param csharp_version:
    :return:
    """
    if csharp_version is None:
        return

    csharp_tag = f"com.unity.ml-agents_{csharp_version}"
    csharp_dirs = ["com.unity.ml-agents", "Project"]
    for csharp_dir in csharp_dirs:
        subprocess.check_call(f"rm -rf {csharp_dir}", shell=True)
        subprocess.check_call(f"git checkout {csharp_tag} -- {csharp_dir}", shell=True)


def undo_git_checkout():
    """
    Clean up the git working directory.
    """
    subprocess.check_call("git reset HEAD .", shell=True)
    subprocess.check_call("git checkout -- .", shell=True)
    # Ensure the cache isn't polluted with old compiled assemblies.
    subprocess.check_call("rm -rf Project/Library", shell=True)


def override_config_file(src_path, dest_path, overrides):
    """
    Override settings in a trainer config file. For example,
        override_config_file(src_path, dest_path, max_steps=42)
    will copy the config file at src_path to dest_path, but override the max_steps field to 42 for all brains.
    """
    with open(src_path) as f:
        configs = yaml.safe_load(f)
        behavior_configs = configs["behaviors"]

    for config in behavior_configs.values():
        _override_config_dict(config, overrides)

    with open(dest_path, "w") as f:
        yaml.dump(configs, f)


def _override_config_dict(config, overrides):
    for key, val in overrides.items():
        if isinstance(val, dict):
            _override_config_dict(config[key], val)
        else:
            config[key] = val


def override_legacy_config_file(python_version, src_path, dest_path, **kwargs):
    """
    Override settings in a trainer config file, using an old version of the src_path. For example,
        override_config_file("0.16.0", src_path, dest_path, max_steps=42)
    will sync the file at src_path from version 0.16.0, copy it to dest_path, and override the
    max_steps field to 42 for all brains.
    """
    # Sync the old version of the file
    python_tag = f"python-packages_{python_version}"
    subprocess.check_call(f"git checkout {python_tag} -- {src_path}", shell=True)

    with open(src_path) as f:
        configs = yaml.safe_load(f)

    for config in configs.values():
        config.update(**kwargs)

    with open(dest_path, "w") as f:
        yaml.dump(configs, f)
