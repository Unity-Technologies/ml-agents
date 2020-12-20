from .communicator import Communicator
from .environment import UnityEnvironment
from mlagents_envs.communicator_objects.unity_rl_output_pb2 import UnityRLOutputProto
from mlagents_envs.communicator_objects.brain_parameters_pb2 import (
    BrainParametersProto,
    ActionSpecProto,
)
from mlagents_envs.communicator_objects.unity_rl_initialization_output_pb2 import (
    UnityRLInitializationOutputProto,
)
from mlagents_envs.communicator_objects.unity_input_pb2 import UnityInputProto
from mlagents_envs.communicator_objects.unity_output_pb2 import UnityOutputProto
from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.observation_pb2 import (
    ObservationProto,
    NONE as COMPRESSION_TYPE_NONE,
    PNG as COMPRESSION_TYPE_PNG,
)


class MockCommunicator(Communicator):
    def __init__(
        self,
        discrete_action=False,
        visual_inputs=0,
        num_agents=3,
        brain_name="RealFakeBrain",
        vec_obs_size=3,
    ):
        """
        Python side of the grpc communication. Python is the client and Unity the server
        """
        super().__init__()
        self.is_discrete = discrete_action
        self.steps = 0
        self.visual_inputs = visual_inputs
        self.has_been_closed = False
        self.num_agents = num_agents
        self.brain_name = brain_name
        self.vec_obs_size = vec_obs_size

    def initialize(self, inputs: UnityInputProto) -> UnityOutputProto:
        if self.is_discrete:
            action_spec = ActionSpecProto(
                num_discrete_actions=2, discrete_branch_sizes=[3, 2]
            )
        else:
            action_spec = ActionSpecProto(num_continuous_actions=2)
        bp = BrainParametersProto(
            brain_name=self.brain_name, is_training=True, action_spec=action_spec
        )
        rl_init = UnityRLInitializationOutputProto(
            name="RealFakeAcademy",
            communication_version=UnityEnvironment.API_VERSION,
            package_version="mock_package_version",
            log_path="",
            brain_parameters=[bp],
        )
        output = UnityRLOutputProto(agentInfos=self._get_agent_infos())
        return UnityOutputProto(rl_initialization_output=rl_init, rl_output=output)

    def _get_agent_infos(self):
        dict_agent_info = {}
        list_agent_info = []
        vector_obs = [1, 2, 3]

        observations = [
            ObservationProto(
                compressed_data=None,
                shape=[30, 40, 3],
                compression_type=COMPRESSION_TYPE_PNG,
            )
            for _ in range(self.visual_inputs)
        ]
        vector_obs_proto = ObservationProto(
            float_data=ObservationProto.FloatData(data=vector_obs),
            shape=[len(vector_obs)],
            compression_type=COMPRESSION_TYPE_NONE,
        )
        observations.append(vector_obs_proto)

        for i in range(self.num_agents):
            list_agent_info.append(
                AgentInfoProto(
                    reward=1,
                    done=(i == 2),
                    max_step_reached=False,
                    id=i,
                    observations=observations,
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
