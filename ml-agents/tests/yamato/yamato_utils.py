import os
import shutil
import subprocess
import yaml
from sys import platform
from typing import List, Optional, Mapping


def get_unity_executable_path():
    if platform == "darwin":
        downloader_install_path = "./.Editor/Unity.app/Contents/MacOS/Unity"
    else:  # if platform == "linux":
        downloader_install_path = "./.Editor/Unity"
    if os.path.exists(downloader_install_path):
        return downloader_install_path
    raise FileNotFoundError("Can't find executable from unity-downloader-cli")


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
    build_target: str = None,
    log_output_path: Optional[str] = f"{get_base_output_path()}/standalone_build.txt",
) -> int:
    """
    Run BuildStandalonePlayerOSX test to produce a player. The location defaults to
    artifacts/standalonebuild/testPlayer.
    """
    unity_exe = get_unity_executable_path()
    print(f"Running BuildStandalonePlayer via {unity_exe}")

    # enum values from https://docs.unity3d.com/2020.3/Documentation/ScriptReference/BuildTarget.html
    build_target_to_enum: Mapping[Optional[str], str] = {
        "mac": "StandaloneOSX",
        "osx": "StandaloneOSX",
        "linux": "StandaloneLinux64",
    }
    # Convert the short name to the official enum
    # Just pass through if it's not on the list.
    build_target_enum = build_target_to_enum.get(build_target, build_target)

    test_args = [
        unity_exe,
        "-projectPath",
        f"{base_path}/Project",
        "-batchmode",
        "-executeMethod",
        "Unity.MLAgents.StandaloneBuildTest.BuildStandalonePlayerOSX",
    ]

    if log_output_path:
        os.makedirs(os.path.dirname(log_output_path), exist_ok=True)
        subprocess.run(["touch", log_output_path])
        test_args += ["-logfile", log_output_path]
    else:
        # Log to stdout
        test_args += ["-logfile", "-"]

    if output_path is not None:
        output_path = os.path.join(get_base_output_path(), output_path)
        test_args += ["--mlagents-build-output-path", output_path]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if scene_path is not None:
        test_args += ["--mlagents-build-scene-path", scene_path]
    if build_target_enum is not None:
        test_args += ["--mlagents-build-target", build_target_enum]
    print(f"{' '.join(test_args)} ...")

    timeout = 30 * 60  # 30 minutes, just in case
    res: subprocess.CompletedProcess = subprocess.run(test_args, timeout=timeout)

    # Copy the default build name into the artifacts folder.
    if output_path is None and res.returncode == 0:
        exe_name = "testPlayer.app" if platform == "darwin" else "testPlayer"
        shutil.move(
            os.path.join(base_path, "Project", exe_name),
            os.path.join(get_base_output_path(), exe_name),
        )

    # Print if we fail or want verbosity.
    if verbose or res.returncode != 0:
        if log_output_path:
            subprocess.run(["cat", log_output_path])

    return res.returncode


def find_executables(root_dir: str) -> List[str]:
    """
    Try to find the player executable. This seems to vary between Unity versions.
    """
    ignored_extension = frozenset([".dll", ".dylib", ".bundle"])
    ignored_files = frozenset(["macblas"])
    exes = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            file_root, ext = os.path.splitext(filename)
            if ext in ignored_extension or filename in ignored_files:
                continue
            file_path = os.path.join(root, filename)
            if os.access(file_path, os.X_OK):
                exes.append(file_path)
    # Also check the input path
    if os.access(root_dir, os.X_OK):
        exes.append(root_dir)
    return exes


def init_venv(
    mlagents_python_version: str = None, extra_packages: Optional[List[str]] = None
) -> None:
    """
    Install the necessary packages for the venv
    :param mlagents_python_version: The version of mlagents python packcage to install.
        If None, will do a local install, otherwise will install from pypi
    :return:
    """
    pip_commands = ["--upgrade pip", "--upgrade setuptools"]
    if mlagents_python_version:
        # install from pypi
        pip_commands += [
            f"mlagents=={mlagents_python_version}",
            f"gym-unity=={mlagents_python_version}",
            # TODO build these and publish to internal pypi
            "tf2onnx==1.6.1",
        ]
    else:
        # Local install
        pip_commands += ["-e ./ml-agents-envs", "-e ./ml-agents", "-e ./gym-unity"]
    if extra_packages:
        pip_commands += extra_packages

    for cmd in pip_commands:
        pip_index_url = "--index-url https://artifactory.prd.it.unity3d.com/artifactory/api/pypi/pypi/simple"
        print(f'Running "python3 -m pip install -q {cmd} {pip_index_url}"')
        subprocess.check_call(
            f"python3 -m pip install -q {cmd} {pip_index_url}", shell=True
        )


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
    csharp_dirs = ["com.unity.ml-agents", "com.unity.ml-agents.extensions", "Project"]
    for csharp_dir in csharp_dirs:
        subprocess.check_call(f"rm -rf {csharp_dir}", shell=True)
        # Allow the checkout to fail, since the extensions folder isn't availabe in 1.0.0
        subprocess.call(f"git checkout {csharp_tag} -- {csharp_dir}", shell=True)


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
