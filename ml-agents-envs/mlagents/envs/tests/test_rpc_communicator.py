import pytest

from mlagents.envs import RpcCommunicator
from mlagents.envs import UnityWorkerInUseException


def test_rpc_communicator_checks_port_on_create():
    first_comm = RpcCommunicator()
    with pytest.raises(UnityWorkerInUseException):
        second_comm = RpcCommunicator()
        second_comm.close()
    first_comm.close()


def test_rpc_communicator_close():
    # Ensures it is possible to open a new RPC Communicators
    # after closing one on the same worker_id
    first_comm = RpcCommunicator()
    first_comm.close()
    second_comm = RpcCommunicator()
    second_comm.close()


def test_rpc_communicator_create_multiple_workers():
    # Ensures multiple RPC communicators can be created with
    # different worker_ids without causing an error.
    first_comm = RpcCommunicator()
    second_comm = RpcCommunicator(worker_id=1)
    first_comm.close()
    second_comm.close()
