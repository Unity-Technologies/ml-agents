from unittest import mock
import pytest
from mlagents_envs.env_utils import validate_environment_path, launch_executable
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.logging_util import (
    set_log_level,
    get_logger,
    INFO,
    ERROR,
    FATAL,
    CRITICAL,
    DEBUG,
)


def mock_glob_method(path):
    """
    Given a path input, returns a list of candidates
    """
    if ".x86" in path:
        return ["linux"]
    if ".app" in path:
        return ["darwin"]
    if ".exe" in path:
        return ["win32"]
    if "*" in path:
        return "Any"
    return []


@mock.patch("sys.platform")
@mock.patch("glob.glob")
def test_validate_path_empty(glob_mock, platform_mock):
    glob_mock.return_value = None
    path = validate_environment_path(" ")
    assert path is None


@mock.patch("mlagents_envs.env_utils.get_platform")
@mock.patch("glob.glob")
def test_validate_path(glob_mock, platform_mock):
    glob_mock.side_effect = mock_glob_method
    for platform in ["linux", "darwin", "win32"]:
        platform_mock.return_value = platform
        path = validate_environment_path(" ")
        assert path == platform


@mock.patch("glob.glob")
@mock.patch("subprocess.Popen")
def test_launch_executable(mock_popen, glob_mock):
    with pytest.raises(UnityEnvironmentException):
        launch_executable(" ", [])
    glob_mock.return_value = ["FakeLaunchPath"]
    launch_executable(" ", [])
    mock_popen.side_effect = PermissionError("Fake permission error")
    with pytest.raises(UnityEnvironmentException):
        launch_executable(" ", [])


def test_set_logging_level():
    for level in [INFO, ERROR, FATAL, CRITICAL, DEBUG]:
        set_log_level(level)
        assert get_logger("test").level == level
