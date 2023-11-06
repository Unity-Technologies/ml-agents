"""
This module provides the 'base_port' pytest fixture for mlagents tests.

This is useful because each mlagents environment requires a unique port to communicate over and will fail on collisions.
Normally this would prevent tests from being run in parallel but with the help of this fixture we can guarantee every
test gets the ports it needs.

See the base_port function for usage details.
"""
import tempfile
from pathlib import Path

import pytest
from filelock import FileLock

# TODO: Use this in all ml-agents tests so they can all run in parallel.
import mlagents.plugins.trainer_type

_BASE_PORT = 6005


# Hook for xdist
# https://github.com/ohmu/pytest-xdist/blob/master/xdist/newhooks.py
def pytest_testnodeready():
    PortAllocator().setup_once_per_node()


class PortAllocator:
    """
    WARNING: Should only be used within this file.
    Handles handing out unique ports to tests that need ports to test.
    Shares state between parallel tests on the same node via a text file and lockfile.
    Should only be used through the base_port test fixture.
    """

    def __init__(self):
        self._port_alloc_file_path: Path = (
            Path(tempfile.gettempdir()) / "next_mla_test_port.txt"
        )
        self._port_alloc_lock_path: Path = self._port_alloc_file_path.with_suffix(
            ".lock"
        )
        self.lock = FileLock(str(self._port_alloc_lock_path))

    def reserve_n_ports(self, n: int) -> int:
        with self.lock:
            if self._port_alloc_file_path.is_file():
                base_port = int(self._port_alloc_file_path.read_text())
            else:
                base_port = 6005
            self._port_alloc_file_path.write_text(str(base_port + n))
        return base_port

    def setup_once_per_node(self) -> None:
        """
        Clean up state files from previous runs, should only be called once per node.
        Intended to only be called via xdist hooks.
        """
        if self._port_alloc_lock_path.exists():
            self._port_alloc_lock_path.unlink(missing_ok=True)
        if self._port_alloc_file_path.exists():
            self._port_alloc_file_path.unlink(missing_ok=True)


@pytest.fixture
def base_port(n_ports: int) -> int:
    """
    Reserve a range of ports for testing (allows parallel testing even with envs).
    Usage:
        @pytest.mark.parametrize("n_ports", [2])
        def test_something(base_port: int) -> None:
            do_something(base_port)
            do_something(base_port + 1)
    :param _port_allocator: The global port allocator (custom pytest fixture).
    :param n_ports: The number of ports needed.
    :return: The base port number.
    """
    return PortAllocator().reserve_n_ports(n_ports)


@pytest.fixture(scope="session", autouse=True)
def setup_plugin_trainers():
    _, _ = mlagents.plugins.trainer_type.register_trainer_plugins()
