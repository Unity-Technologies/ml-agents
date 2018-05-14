import logging
import grpc

from multiprocessing import Pipe
from concurrent.futures import ThreadPoolExecutor

from .communicator import Communicator
from communicator_objects import UnityToExternalServicer, add_UnityToExternalServicer_to_server
from communicator_objects import UnityMessage, UnityInput, UnityOutput
from .exception import UnityTimeOutException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unityagents")


class UnityToExternalServicerImplementation(UnityToExternalServicer):
    parent_conn, child_conn = Pipe()

    def Initialize(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()

    def Exchange(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()


class RpcCommunicator(Communicator):
    def __init__(self, worker_id=0,
                 base_port=5005):
        """
        Python side of the grpc communication. Python is the server and Unity the client


        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """
        self.port = base_port + worker_id
        self.worker_id = worker_id
        self.server = None
        self.unity_to_external = None
        self.is_open = False

    def initialize(self, inputs: UnityInput) -> UnityOutput:
        try:
            # Establish communication grpc
            self.server = grpc.server(ThreadPoolExecutor(max_workers=10))
            self.unity_to_external = UnityToExternalServicerImplementation()
            add_UnityToExternalServicer_to_server(self.unity_to_external, self.server)
            self.server.add_insecure_port('[::]:'+str(self.port))
            self.server.start()
        except :
            raise UnityTimeOutException(
                "Couldn't start socket communication because worker number {} is still in use. "
                "You may need to manually close a previously opened environment "
                "or use a different worker number.".format(str(self.worker_id)))
        if not self.unity_to_external.parent_conn.poll(30):
            raise UnityTimeOutException(
                "The Unity environment took too long to respond. Make sure that :\n"
                "\t The environment does not need user interaction to launch\n"
                "\t The Academy and the External Brain(s) are attached to objects in the Scene\n"
                "\t The environment and the Python interface have compatible versions.")
        aca_param = self.unity_to_external.parent_conn.recv().unity_output
        self.is_open = True
        message = UnityMessage()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        self.unity_to_external.parent_conn.recv()
        return aca_param

    def exchange(self, inputs: UnityInput) -> UnityOutput:
        message = UnityMessage()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        output = self.unity_to_external.parent_conn.recv()
        if output.header.status != 200:
            return None
        return output.unity_output

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the grpc connection.
        """
        if self.is_open:
            message_input = UnityMessage()
            message_input.header.status = 400
            self.unity_to_external.parent_conn.send(message_input)
            self.unity_to_external.parent_conn.close()
            self.server.stop(False)
            self.is_open = False




