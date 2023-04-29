import tempfile
from pathlib import Path

import pytest
from filelock import FileLock

import mlagents.plugins.trainer_type

class PortAllocator:
    """
    Handles handing out unique ports to tests that need ports to test.
    Shares state between parallel tests on the same node via a text file and lockfile.
    Should only be used through the base_port test fixture.
    """

    def __init__(self, base_port=6005):
        """
        Initializes a new PortAllocator object.

        Args:
            base_port (int): The starting port number for port allocation.
        """
        self._base_port = base_port
        self._port_alloc_file_path = Path(tempfile.gettempdir()) / "next_mla_test_port.txt"
        self._port_alloc_lock_path = self._port_alloc_file_path.with_suffix(".lock")
        self.lock = FileLock(str(self._port_alloc_lock_path))

    def reserve_ports(self, n_ports):
        """
        Reserves a range of ports for testing.

        Args:
            n_ports (int): The number of ports to reserve.

        Returns:
            int: The base port number.
        """
        with self.lock:
            if self._port_alloc_file_path.is_file():
                base_port = int(self._port_alloc_file_path.read_text())
            else:
                base_port = self._base_port
            self._port_alloc_file_path.write_text(str(base_port + n_ports))
        return base_port

    def setup_once_per_node(self):
        """
        Cleans up state files from previous runs.
        Should only be called once per node.
        """
        self._port_alloc_lock_path.unlink(missing_ok=True)
        self._port_alloc_file_path.unlink(missing_ok=True)

@pytest.fixture
def base_port():
    """
    Reserve a range of ports for testing (allows parallel testing even with envs).
    Usage:
        @pytest.mark.parametrize("n_ports", [2])
        def test_something(base_port: int) -> None:
            do_something(base_port)
            do_something(base_port + 1)

    Returns:
        int: The base port number.
    """
    port_allocator = PortAllocator()
    return port_allocator.reserve_ports(n_ports=1)

@pytest.fixture(scope="session", autouse=True)
def setup_plugin_trainers():
    _, _ = mlagents.plugins.trainer_type.register_trainer_plugins()
