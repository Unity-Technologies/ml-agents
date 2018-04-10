import logging
import numpy as np
import time


from communicator import PythonToUnityStub
from communicator import UnityOutput, UnityInput, PythonParameters, AgentAction

import grpc

from .brain import BrainInfo, BrainParameters, AllBrainInfo
from .exception import UnityActionException, UnityTimeOutException


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
        self._buffer_size = 12000
        self._stub = None
        self._log_path = None
        self._brain_names = None
        self._brains = None
        self._stub = None
        self.channel = None

        try:
            # Establish communication socket
            self.channel = grpc.insecure_channel('localhost:' + str(port))
            # self._stub = PythonToUnityStub(channel)
        except :
            raise UnityTimeOutException("Couldn't start socket communication because worker number {} is still in use. "
                               "You may need to manually close a previously opened environment "
                               "or use a different worker number.".format(str(worker_id)))

    def get_academy_parameters(self):
        # This must take in PythonParameters
        try:
            grpc.channel_ready_future(self.channel).result(timeout=30)
        except grpc.FutureTimeoutError:
            raise UnityTimeOutException("gRPC Timeout")
        else:
            self._stub = PythonToUnityStub(self.channel)
        print("Client Connected")

        # Put the seed and the logpath here
        aca_params = self._stub.Initialize(PythonParameters(), )
        p = {}
        self._brain_names = []
        self._brains = {}
        external_brain_names = []
        for brain_param in aca_params.brain_parameters:
            self._brain_names += [brain_param.brain_name]
            self._brains[brain_param.brain_name] = \
                BrainParameters(brain_param.brain_name, {
                    "vectorObservationSize": brain_param.vector_observation_size,
                    "numStackedVectorObservations": brain_param.num_stacked_vector_observations,
                    "cameraResolutions": brain_param.camera_resolutions,
                    "vectorActionSize": brain_param.vector_action_size,
                    "vectorActionDescriptions": brain_param.vector_action_descriptions,
                    "vectorActionSpaceType": brain_param.vector_action_space_type,
                    "vectorObservationSpaceType": brain_param.vector_observation_space_type
                  })
            if brain_param.brain_type == 2:
                external_brain_names += [brain_param.brain_name]
        self._log_path = ''  # TODO : Change

        p["apiNumber"] = aca_params.version
        p["AcademyName"] = aca_params.name
        p["logPath"] = self._log_path
        p["brainNames"] = self._brain_names
        p["externalBrainNames"] = external_brain_names
        p["brainParameters"] = self._brains
        p["resetParameters"] = {}
        for k in p :
            print("\n" + k +" -> "+str(p[k]))
        return p

    # def call_reset(self, train_mode, config):
    #     outputs = self._stub.Send(UnityInput())
    #
    #     return self._get_state(outputs)
    #
    # def call_step(self, vector_action, memory, text_action):
    #     outputs = self._stub.Send(self._generate_input(vector_action, memory, text_action))
    #     return self._get_state(outputs)

    def send(self, inputs: UnityInput) -> UnityOutput:
        outputs = self._stub.Send(inputs)
        return outputs




    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        # if self._conn is not None:
        #     self._conn.send(b"EXIT")
        #     self._conn.close()
        #     self._socket.close()
        #     self._conn = None

