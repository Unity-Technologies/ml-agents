import grpc
from typing import Optional

from sys import platform
import socket
from multiprocessing import Pipe
from concurrent.futures import ThreadPoolExecutor

from .communicator import Communicator
from mlagents_envs.communicator_objects.unity_to_external_pb2_grpc import (
    UnityToExternalProtoServicer,
    add_UnityToExternalProtoServicer_to_server,
)
from mlagents_envs.communicator_objects.unity_message_pb2 import UnityMessageProto
from mlagents_envs.communicator_objects.unity_input_pb2 import UnityInputProto
from mlagents_envs.communicator_objects.unity_output_pb2 import UnityOutputProto
from .exception import UnityTimeOutException, UnityWorkerInUseException


class UnityToExternalServicerImplementation(UnityToExternalProtoServicer):
    def __init__(self):
        self.parent_conn, self.child_conn = Pipe()

    def Initialize(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()

    def Exchange(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()


class RpcCommunicator(Communicator):
    def __init__(self, worker_id=0, base_port=5005, timeout_wait=30):
        """
        Python side of the grpc communication. Python is the server and Unity the client


        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Offset from base_port. Used for training multiple environments simultaneously.
        :int timeout_wait: Timeout (in seconds) to wait for a response before exiting.
        """
        super().__init__(worker_id, base_port)
        self.port = base_port + worker_id
        self.worker_id = worker_id
        self.timeout_wait = timeout_wait
        self.server = None
        self.unity_to_external = None
        self.is_open = False
        self.create_server()

    def create_server(self):
        """
        Creates the GRPC server.
        """
        self.check_port(self.port)

        try:
            # Establish communication grpc
            self.server = grpc.server(ThreadPoolExecutor(max_workers=10))
            self.unity_to_external = UnityToExternalServicerImplementation()
            add_UnityToExternalProtoServicer_to_server(
                self.unity_to_external, self.server
            )
            # Using unspecified address, which means that grpc is communicating on all IPs
            # This is so that the docker container can connect.
            self.server.add_insecure_port("[::]:" + str(self.port))
            self.server.start()
            self.is_open = True
        except Exception:
            raise UnityWorkerInUseException(self.worker_id)

    def check_port(self, port):
        """
        Attempts to bind to the requested communicator port, checking if it is already in use.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if platform == "linux" or platform == "linux2":
            # On linux, the port remains unusable for TIME_WAIT=60 seconds after closing
            # SO_REUSEADDR frees the port right after closing the environment
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("localhost", port))
        except OSError:
            raise UnityWorkerInUseException(self.worker_id)
        finally:
            s.close()

    def poll_for_timeout(self):
        """
        Polls the GRPC parent connection for data, to be used before calling recv.  This prevents
        us from hanging indefinitely in the case where the environment process has died or was not
        launched.
        """
        if not self.unity_to_external.parent_conn.poll(self.timeout_wait):
            raise UnityTimeOutException(
                "The Unity environment took too long to respond. Make sure that :\n"
                "\t The environment does not need user interaction to launch\n"
                '\t The Agents\' Behavior Parameters > Behavior Type is set to "Default"\n'
                "\t The environment and the Python interface have compatible versions."
            )

    def initialize(self, inputs: UnityInputProto) -> UnityOutputProto:
        self.poll_for_timeout()
        aca_param = self.unity_to_external.parent_conn.recv().unity_output
        message = UnityMessageProto()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        self.unity_to_external.parent_conn.recv()
        return aca_param

    def exchange(self, inputs: UnityInputProto) -> Optional[UnityOutputProto]:
        message = UnityMessageProto()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        self.poll_for_timeout()
        output = self.unity_to_external.parent_conn.recv()
        if output.header.status != 200:
            return None
        return output.unity_output

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the grpc connection.
        """
        if self.is_open:
            message_input = UnityMessageProto()
            message_input.header.status = 400
            self.unity_to_external.parent_conn.send(message_input)
            self.unity_to_external.parent_conn.close()
            self.server.stop(False)
            self.is_open = False
