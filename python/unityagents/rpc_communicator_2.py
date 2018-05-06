import logging
import grpc

from multiprocessing import Pipe
from concurrent.futures import ThreadPoolExecutor

from .communicator import Communicator
from communicator_objects import UnityToPythonServicer, add_UnityToPythonServicer_to_server
from communicator_objects import UnityRLOutput, UnityRLInput,\
    UnityInput, UnityOutput, AcademyParameters, \
    UnityInitializationOutput, UnityInitializationInput
from .exception import UnityTimeOutException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unityagents")


class Unity2Python(UnityToPythonServicer):
    parent_conn, child_conn = Pipe()

    def Initialize(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()

    def Send(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()


class RpcCommunicator2(Communicator):
    def __init__(self, worker_id=0,
                 base_port=5005):
        """
        Python side of the grpc communication. Python is the server and Unity the client


        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """

        port = base_port + worker_id
        self._empty_message = UnityInput()
        self._empty_message.header.status = 200

        try:
            # Establish communication grpc
            self.server = grpc.server(ThreadPoolExecutor(max_workers=10))
            self.unity_to_python = Unity2Python()
            add_UnityToPythonServicer_to_server(self.unity_to_python, self.server)
            self.server.add_insecure_port('[::]:'+str(port))
            self.server.start()
            print("Server Started")
        except :
            raise UnityTimeOutException("Couldn't start socket communication because worker number {} is still in use. "
                               "You may need to manually close a previously opened environment "
                               "or use a different worker number.".format(str(worker_id)))

    def get_academy_parameters(self, python_parameters) -> AcademyParameters:
        aca_param = self.unity_to_python.parent_conn.recv().academy_parameters
        initialization_inputs = UnityInitializationInput()
        initialization_inputs.header.status = 200
        initialization_inputs.python_parameters.CopyFrom(python_parameters)
        self.unity_to_python.parent_conn.send(initialization_inputs)
        self.unity_to_python.parent_conn.recv()
        return aca_param

    def send(self, inputs: UnityRLInput) -> UnityRLOutput:
        self._empty_message.rl_input.CopyFrom(inputs)
        self.unity_to_python.parent_conn.send(self._empty_message)
        output = self.unity_to_python.parent_conn.recv()
        if output.header.status == 400:
            raise KeyboardInterrupt
        return output.rl_output

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the grpc connection.
        """
        try:
            message_input = UnityInput()
            message_input.header.status = 400
            self.unity_to_python.parent_conn.send(message_input)
        except :
            pass


