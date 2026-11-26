import subprocess
import sys
from textwrap import dedent


def _run_in_fresh_interpreter(source: str) -> None:
    # A cached package import would hide calls made by its initializer.
    subprocess.run([sys.executable, "-c", dedent(source)], check=True)


def test_wrapper_imports_work_offline() -> None:
    _run_in_fresh_interpreter(
        """
        from unittest.mock import patch
        from urllib.error import URLError

        with patch(
            "mlagents_envs.registry.unity_env_registry.load_remote_manifest",
            side_effect=URLError("offline"),
        ) as load_manifest:
            from mlagents_envs.envs.unity_aec_env import UnityAECEnv
            from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
            from mlagents_envs.envs import unity_parallel_env

            assert unity_parallel_env.UnityParallelEnv
            load_manifest.assert_not_called()
        """
    )


def test_named_examples_load_from_registry_on_demand() -> None:
    _run_in_fresh_interpreter(
        """
        from unittest.mock import patch

        entry = {
            "expected_reward": None,
            "description": None,
            "linux_url": None,
            "darwin_url": None,
            "win_url": None,
        }
        manifest = {
            "environments": [
                {"StrikersVsGoalie": entry.copy()},
                {"3DBall": entry.copy()},
            ]
        }
        with patch(
            "mlagents_envs.registry.unity_env_registry.load_remote_manifest",
            return_value=manifest,
        ) as load_manifest:
            import mlagents_envs.envs as envs

            load_manifest.assert_not_called()
            from mlagents_envs.envs import StrikersVsGoalie, ThreeDBall

            assert StrikersVsGoalie.env_id == "StrikersVsGoalie"
            assert ThreeDBall.env_id == "3DBall"
            assert envs.StrikersVsGoalie is StrikersVsGoalie
            assert "ThreeDBall" in dir(envs)
            namespace = {}
            exec("from mlagents_envs.envs import *", namespace)
            assert namespace["ThreeDBall"] is ThreeDBall
            load_manifest.assert_called_once()
        """
    )
