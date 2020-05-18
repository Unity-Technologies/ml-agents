import shutil

from mlagents_envs.registry import default_registry
from mlagents_envs.registry.binary_utils import get_tmp_dir

BASIC_ID = "Basic"


def delete_binaries():
    tmp_dir, bin_dir = get_tmp_dir()
    shutil.rmtree(tmp_dir)
    shutil.rmtree(bin_dir)


def test_basic_in_registry():
    delete_binaries()
    for worker_id in range(2):
        assert BASIC_ID in default_registry
        env = default_registry[BASIC_ID].make(worker_id=worker_id)
        env.reset()
        env.step()
        assert len(env.behavior_specs) == 1
        env.close()
