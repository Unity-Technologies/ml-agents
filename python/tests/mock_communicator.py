
from unityagents.communicator import Communicator
from communicator_objects import UnityMessage, UnityOutput, UnityInput,\
    ResolutionProto, BrainParametersProto, UnityRLInitializationOutput,\
    AgentInfoProto, UnityRLOutput


class MockCommunicator(Communicator):
    def __init__(self, discrete_action=False, visual_inputs=0):
        """
        Python side of the grpc communication. Python is the client and Unity the server

        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """
        self.is_discrete = discrete_action
        self.steps = 0
        self.visual_inputs = visual_inputs
        self.has_been_closed = False

    def initialize(self, inputs: UnityInput) -> UnityOutput:
        resolutions = [ResolutionProto(
            width=30,
            height=40,
            gray_scale=False) for i in range(self.visual_inputs)]
        bp = BrainParametersProto(
            vector_observation_size=3,
            num_stacked_vector_observations=2,
            vector_action_size=2,
            camera_resolutions=resolutions,
            vector_action_descriptions=["", ""],
            vector_action_space_type=int(not self.is_discrete),
            vector_observation_space_type=1,
            brain_name="RealFakeBrain",
            brain_type=2
        )
        rl_init = UnityRLInitializationOutput(
            name="RealFakeAcademy",
            version="API-4",
            log_path="",
            brain_parameters=[bp]
        )
        return UnityOutput(
            rl_initialization_output=rl_init
        )

    def exchange(self, inputs: UnityInput) -> UnityOutput:
        dict_agent_info = {}
        if self.is_discrete:
            vector_action = [1]
        else:
            vector_action = [1, 2]
        list_agent_info = []
        for i in range(3):
            list_agent_info.append(
                AgentInfoProto(
                    stacked_vector_observation=[1, 2, 3, 1, 2, 3],
                    reward=1,
                    stored_vector_actions=vector_action,
                    stored_text_actions="",
                    text_observation="",
                    memories=[],
                    done=(i == 2),
                    max_step_reached=False,
                    id=i
                ))
        dict_agent_info["RealFakeBrain"] = \
            UnityRLOutput.ListAgentInfoProto(value=list_agent_info)
        global_done = False
        try:
            global_done = (inputs.rl_input.agent_actions["RealFakeBrain"].value[0].vector_actions[0] == -1)
        except:
            pass
        result = UnityRLOutput(
            global_done=global_done,
            agentInfos=dict_agent_info
        )
        return UnityOutput(
            rl_output=result
        )

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the grpc connection.
        """
        self.has_been_closed = True
