from .communicator import Communicator
from .environment import UnityEnvironment
from mlagents.envs.communicator_objects.unity_rl_output_pb2 import UnityRLOutputProto
from mlagents.envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents.envs.communicator_objects.unity_rl_initialization_output_pb2 import (
    UnityRLInitializationOutputProto,
)
from mlagents.envs.communicator_objects.unity_input_pb2 import UnityInputProto
from mlagents.envs.communicator_objects.unity_output_pb2 import UnityOutputProto
from mlagents.envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents.envs.communicator_objects.compressed_observation_pb2 import (
    CompressedObservationProto,
    CompressionTypeProto,
)


class MockCommunicator(Communicator):
    def __init__(
        self,
        discrete_action=False,
        visual_inputs=0,
        stack=True,
        num_agents=3,
        brain_name="RealFakeBrain",
        vec_obs_size=3,
    ):
        """
        Python side of the grpc communication. Python is the client and Unity the server

        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """
        self.is_discrete = discrete_action
        self.steps = 0
        self.visual_inputs = visual_inputs
        self.has_been_closed = False
        self.num_agents = num_agents
        self.brain_name = brain_name
        self.vec_obs_size = vec_obs_size
        if stack:
            self.num_stacks = 2
        else:
            self.num_stacks = 1

    def initialize(self, inputs: UnityInputProto) -> UnityOutputProto:
        bp = BrainParametersProto(
            vector_observation_size=self.vec_obs_size,
            num_stacked_vector_observations=self.num_stacks,
            vector_action_size=[2],
            vector_action_descriptions=["", ""],
            vector_action_space_type=int(not self.is_discrete),
            brain_name=self.brain_name,
            is_training=True,
        )
        rl_init = UnityRLInitializationOutputProto(
            name="RealFakeAcademy",
            version=UnityEnvironment.API_VERSION,
            log_path="",
            brain_parameters=[bp],
        )
        output = UnityRLOutputProto(agentInfos=self._get_agent_infos())
        return UnityOutputProto(rl_initialization_output=rl_init, rl_output=output)

    def _get_agent_infos(self):
        dict_agent_info = {}
        if self.is_discrete:
            vector_action = [1]
        else:
            vector_action = [1, 2]
        list_agent_info = []
        if self.num_stacks == 1:
            observation = [1, 2, 3]
        else:
            observation = [1, 2, 3, 1, 2, 3]

        compressed_obs = [
            CompressedObservationProto(
                data=None, shape=[30, 40, 3], compression_type=CompressionTypeProto.PNG
            )
            for _ in range(self.visual_inputs)
        ]

        for i in range(self.num_agents):
            list_agent_info.append(
                AgentInfoProto(
                    stacked_vector_observation=observation,
                    reward=1,
                    stored_vector_actions=vector_action,
                    stored_text_actions="",
                    text_observation="",
                    memories=[],
                    done=(i == 2),
                    max_step_reached=False,
                    id=i,
                    compressed_observations=compressed_obs,
                )
            )
        dict_agent_info["RealFakeBrain"] = UnityRLOutputProto.ListAgentInfoProto(
            value=list_agent_info
        )
        return dict_agent_info

    def exchange(self, inputs: UnityInputProto) -> UnityOutputProto:
        result = UnityRLOutputProto(agentInfos=self._get_agent_infos())
        return UnityOutputProto(rl_output=result)

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the grpc connection.
        """
        self.has_been_closed = True
