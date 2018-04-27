import logging
import socket
import struct

from .communicator import Communicator
from communicator import UnityRLOutput, UnityRLInput,\
    UnityOutput, UnityInput, AcademyParameters,\
    UnityInitializationInput, UnityInitializationOutput
from .exception import UnityTimeOutException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unityagents")


class SocketCommunicator(Communicator):
    def __init__(self, worker_id=0,
                 base_port=5005):
        """
        Python side of the socket communication

        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """

        port = base_port + worker_id
        self._buffer_size = 12000
        self._empty_message = UnityInput()
        self._empty_message.header.status = 200

        try:
            # Establish communication socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("localhost", port))
            self._socket.settimeout(30)
            self._socket.listen(1)
            self._conn, _ = self._socket.accept()
            self._conn.settimeout(30)
            # self._stub = PythonToUnityStub(channel)
        except :
            raise UnityTimeOutException("Couldn't start socket communication because worker number {} is still in use. "
                               "You may need to manually close a previously opened environment "
                               "or use a different worker number.".format(str(worker_id)))

    def get_academy_parameters(self, python_parameters) -> AcademyParameters:
        initialization_input = UnityInitializationInput()
        initialization_input.header.status = 200
        initialization_input.python_parameters.CopyFrom(python_parameters)
        self._communicator_send(initialization_input.SerializeToString())
        initialization_output = UnityInitializationOutput()
        initialization_output.ParseFromString(self._communicator_receive())
        return initialization_output.academy_parameters

    def _communicator_receive(self):
        try:
            s = self._conn.recv(self._buffer_size)
            message_length = struct.unpack("I", bytearray(s[:4]))[0]
            s = s[4:]
            while len(s) != message_length:
                s += self._conn.recv(self._buffer_size)
        except socket.timeout as e:
            raise UnityTimeOutException("The environment took too long to respond.")
        return s

    def _communicator_send(self, message):
        self._conn.send(struct.pack("I", len(message)) + message)

    def send(self, inputs: UnityRLInput) -> UnityRLOutput:
        message_input = self._empty_message
        message_input.rl_input.CopyFrom(inputs)
        self._communicator_send(message_input.SerializeToString())
        result = UnityOutput()
        result.ParseFromString(self._communicator_receive())
        return result.rl_output

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        # inputs = UnityInput()
        # inputs.command = 2
        # How to shut gRPC down ?
        try:
            message_input = UnityInput()
            message_input.header.status = 400
            self._communicator_send(message_input.SerializeToString())
        except :
            pass
        # if self._conn is not None:
        #     self._conn.send(b"EXIT")
        #     self._conn.close()
        #     self._socket.close()
        #     self._conn = None

