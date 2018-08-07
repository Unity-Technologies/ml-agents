import logging
import socket
import struct

from .communicator import Communicator
from communicator_objects import UnityMessage, UnityOutput, UnityInput
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

        self.port = base_port + worker_id
        self._buffer_size = 12000
        self.worker_id = worker_id
        self._socket = None
        self._conn = None

    def initialize(self, inputs: UnityInput) -> UnityOutput:
        try:
            # Establish communication socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("localhost", self.port))
        except:
            raise UnityTimeOutException("Couldn't start socket communication because worker number {} is still in use. "
                                        "You may need to manually close a previously opened environment "
                                        "or use a different worker number.".format(str(self.worker_id)))
        try:
            self._socket.settimeout(30)
            self._socket.listen(1)
            self._conn, _ = self._socket.accept()
            self._conn.settimeout(30)
        except :
            raise UnityTimeOutException(
                "The Unity environment took too long to respond. Make sure that :\n"
                "\t The environment does not need user interaction to launch\n"
                "\t The Academy and the External Brain(s) are attached to objects in the Scene\n"
                "\t The environment and the Python interface have compatible versions.")
        message = UnityMessage()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self._communicator_send(message.SerializeToString())
        initialization_output = UnityMessage()
        initialization_output.ParseFromString(self._communicator_receive())
        return initialization_output.unity_output

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

    def exchange(self, inputs: UnityInput) -> UnityOutput:
        message = UnityMessage()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self._communicator_send(message.SerializeToString())
        outputs = UnityMessage()
        outputs.ParseFromString(self._communicator_receive())
        if outputs.header.status != 200:
            return None
        return outputs.unity_output

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._socket is not None and self._conn is not None:
            message_input = UnityMessage()
            message_input.header.status = 400
            self._communicator_send(message_input.SerializeToString())
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._socket is not None:
            self._conn.close()
            self._conn = None

