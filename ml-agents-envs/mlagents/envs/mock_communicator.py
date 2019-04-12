from .communicator import Communicator
from .communicator_objects import UnityOutput, UnityInput, \
    ResolutionProto, BrainParametersProto, UnityRLInitializationOutput, \
    AgentInfoProto, UnityRLOutput


class MockCommunicator(Communicator):
    def __init__(self, discrete_action=False, visual_inputs=0, stack=True, num_agents=3,
                 brain_name="RealFakeBrain", vec_obs_size=3):
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

    def initialize(self, inputs: UnityInput) -> UnityOutput:
        resolutions = [ResolutionProto(
            width=30,
            height=40,
            gray_scale=False) for i in range(self.visual_inputs)]
        bp = BrainParametersProto(
            vector_observation_size=self.vec_obs_size,
            num_stacked_vector_observations=self.num_stacks,
            vector_action_size=[2],
            camera_resolutions=resolutions,
            vector_action_descriptions=["", ""],
            vector_action_space_type=int(not self.is_discrete),
            brain_name=self.brain_name,
            is_training=True
        )
        rl_init = UnityRLInitializationOutput(
            name="RealFakeAcademy",
            version="API-8",
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
        if self.num_stacks == 1:
            observation = [1, 2, 3]
        else:
            observation = [1, 2, 3, 1, 2, 3]

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
                    id=i
                ))
        dict_agent_info["RealFakeBrain"] = \
            UnityRLOutput.ListAgentInfoProto(value=list_agent_info)
        global_done = False
        try:
            fake_brain = inputs.rl_input.agent_actions["RealFakeBrain"]
            global_done = (fake_brain.value[0].vector_actions[0] == -1)
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
