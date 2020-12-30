import shutil
import os

from mlagents_envs.registry import default_registry, UnityEnvRegistry
from mlagents_envs.registry.remote_registry_entry import RemoteRegistryEntry
from mlagents_envs.registry.binary_utils import get_tmp_dir

BASIC_ID = "Basic"


def delete_binaries():
    tmp_dir, bin_dir = get_tmp_dir()
    shutil.rmtree(tmp_dir)
    shutil.rmtree(bin_dir)


def create_registry():
    reg = UnityEnvRegistry()
    entry = RemoteRegistryEntry(
        BASIC_ID,
        0.0,
        "Description",
        "https://storage.googleapis.com/mlagents-test-environments/1.0.0/linux/Basic.zip",
        "https://storage.googleapis.com/mlagents-test-environments/1.0.0/darwin/Basic.zip",
        "https://storage.googleapis.com/mlagents-test-environments/1.0.0/windows/Basic.zip",
    )
    reg.register(entry)
    return reg


def test_basic_in_registry():
    assert BASIC_ID in default_registry
    os.environ["TERM"] = "xterm"
    delete_binaries()
    registry = create_registry()
    for worker_id in range(2):
        assert BASIC_ID in registry
        env = registry[BASIC_ID].make(
            base_port=6002, worker_id=worker_id, no_graphics=True
        )
        env.reset()
        env.step()
        assert len(env.behavior_specs) == 1
        env.close()
