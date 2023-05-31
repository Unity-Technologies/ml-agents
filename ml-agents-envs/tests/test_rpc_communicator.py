from unittest.mock import Mock

import pytest
from unittest import mock

import grpc

import mlagents_envs.rpc_communicator
from mlagents_envs.rpc_communicator import RpcCommunicator
from mlagents_envs.exception import (
    UnityWorkerInUseException,
    UnityTimeOutException,
    UnityEnvironmentException,
)
from mlagents_envs.communicator_objects.unity_input_pb2 import UnityInputProto


@pytest.mark.parametrize("n_ports", [1])
def test_rpc_communicator_checks_port_on_create(base_port: int) -> None:
    first_comm = RpcCommunicator(base_port=base_port)
    with pytest.raises(UnityWorkerInUseException):
        second_comm = RpcCommunicator(base_port=base_port)
        second_comm.close()
    first_comm.close()


@pytest.mark.parametrize("n_ports", [2])
def test_rpc_communicator_close(base_port: int) -> None:
    # Ensures it is possible to open a new RPC Communicators
    # after closing one on the same worker_id
    first_comm = RpcCommunicator(base_port=base_port)
    first_comm.close()
    second_comm = RpcCommunicator(base_port=base_port + 1)
    second_comm.close()


@pytest.mark.parametrize("n_ports", [2])
def test_rpc_communicator_create_multiple_workers(base_port: int) -> None:
    # Ensures multiple RPC communicators can be created with
    # different worker_ids without causing an error.
    first_comm = RpcCommunicator(base_port=base_port)
    second_comm = RpcCommunicator(base_port=base_port, worker_id=1)
    first_comm.close()
    second_comm.close()


@pytest.mark.parametrize("n_ports", [1])
@mock.patch.object(grpc, "server")
@mock.patch.object(
    mlagents_envs.rpc_communicator, "UnityToExternalServicerImplementation"
)
def test_rpc_communicator_initialize_OK(
    mock_impl: Mock, mock_grpc_server: Mock, base_port: int
) -> None:
    comm = RpcCommunicator(base_port=base_port, timeout_wait=0.25)
    comm.unity_to_external.parent_conn.poll.return_value = True
    input = UnityInputProto()
    comm.initialize(input)
    comm.unity_to_external.parent_conn.poll.assert_called()


@pytest.mark.parametrize("n_ports", [1])
@mock.patch.object(grpc, "server")
@mock.patch.object(
    mlagents_envs.rpc_communicator, "UnityToExternalServicerImplementation"
)
def test_rpc_communicator_initialize_timeout(
    mock_impl: Mock, mock_grpc_server: Mock, base_port: int
) -> None:
    comm = RpcCommunicator(timeout_wait=0.25, base_port=base_port)
    comm.unity_to_external.parent_conn.poll.return_value = None
    input = UnityInputProto()
    # Expect a timeout
    with pytest.raises(UnityTimeOutException):
        comm.initialize(input)
    comm.unity_to_external.parent_conn.poll.assert_called()


@pytest.mark.parametrize("n_ports", [1])
@mock.patch.object(grpc, "server")
@mock.patch.object(
    mlagents_envs.rpc_communicator, "UnityToExternalServicerImplementation"
)
def test_rpc_communicator_initialize_callback(
    mock_impl: Mock, mock_grpc_server: Mock, base_port: int
) -> None:
    def callback():
        raise UnityEnvironmentException

    comm = RpcCommunicator(base_port=base_port, timeout_wait=0.25)
    comm.unity_to_external.parent_conn.poll.return_value = None
    input = UnityInputProto()
    # Expect a timeout
    with pytest.raises(UnityEnvironmentException):
        comm.initialize(input, poll_callback=callback)
    comm.unity_to_external.parent_conn.poll.assert_called()
