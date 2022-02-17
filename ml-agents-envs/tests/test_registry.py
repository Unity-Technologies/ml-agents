import os
from pathlib import Path

import pytest

from mlagents_envs.registry import default_registry, UnityEnvRegistry
from mlagents_envs.registry.remote_registry_entry import RemoteRegistryEntry

BASIC_ID = "Basic"


def create_registry(tmp_dir: str) -> UnityEnvRegistry:
    reg = UnityEnvRegistry()
    entry = RemoteRegistryEntry(
        BASIC_ID,
        0.0,
        "Description",
        "https://storage.googleapis.com/mlagents-test-environments/1.0.0/linux/Basic.zip",
        "https://storage.googleapis.com/mlagents-test-environments/1.0.0/darwin/Basic.zip",
        "https://storage.googleapis.com/mlagents-test-environments/1.0.0/windows/Basic.zip",
        tmp_dir=tmp_dir,
    )
    reg.register(entry)
    return reg


@pytest.mark.parametrize("n_ports", [2])
def test_basic_in_registry(base_port: int, tmp_path: Path) -> None:
    assert BASIC_ID in default_registry
    os.environ["TERM"] = "xterm"
    registry = create_registry(str(tmp_path))
    for worker_id in range(2):
        assert BASIC_ID in registry
        env = registry[BASIC_ID].make(
            base_port=base_port, worker_id=worker_id, no_graphics=True
        )
        env.reset()
        env.step()
        assert len(env.behavior_specs) == 1
        env.close()
