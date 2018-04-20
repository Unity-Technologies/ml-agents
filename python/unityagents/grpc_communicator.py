import logging
import grpc

from communicator import PythonToUnityStub
from communicator import UnityRLOutput, UnityRLInput,\
    UnityInput, AcademyParameters, UnityInitializationInput
from .exception import UnityTimeOutException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unityagents")


class GrpcCommunicator(object):
    def __init__(self, worker_id=0,
                 base_port=5005):
        """
        Python side of the socket communication

        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """

        port = base_port + worker_id
        self._stub = None
        self.channel = None
        self._empty_message = UnityInput()
        self._empty_message.header.status = 200

        try:
            # Establish communication grpc
            self.channel = grpc.insecure_channel('localhost:' + str(port))
        except :
            raise UnityTimeOutException("Couldn't start socket communication because worker number {} is still in use. "
                               "You may need to manually close a previously opened environment "
                               "or use a different worker number.".format(str(worker_id)))

    def get_academy_parameters(self, python_parameters) -> AcademyParameters:
        try:
            grpc.channel_ready_future(self.channel).result(timeout=30)
        except grpc.FutureTimeoutError:
            raise UnityTimeOutException("gRPC Timeout")
        else:
            self._stub = PythonToUnityStub(self.channel)
        initialization_input = UnityInitializationInput()
        initialization_input.header.status = 200
        initialization_input.python_parameters.CopyFrom(python_parameters)

        # Put the seed and the logpath here
        initialization_output = self._stub.Initialize(initialization_input, )
        return initialization_output.academy_parameters

    def send(self, inputs: UnityRLInput) -> UnityRLOutput:
        message_input = self._empty_message
        message_input.rl_input.CopyFrom(inputs)
        outputs = self._stub.Send(message_input)
        return outputs.rl_output

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
            self._stub.Send(message_input)
        except :
            pass
        # if self._conn is not None:
        #     self._conn.send(b"EXIT")
        #     self._conn.close()
        #     self._socket.close()
        #     self._conn = None

