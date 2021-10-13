import atexit
from distutils.version import StrictVersion

import numpy as np
import os
import subprocess
from typing import Dict, List, Optional, Tuple, Mapping as MappingType

import mlagents_envs

from mlagents_envs.logging_util import get_logger
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel import DefaultTrainingAnalyticsSideChannel
from mlagents_envs.side_channel.side_channel_manager import SideChannelManager
from mlagents_envs import env_utils

from mlagents_envs.base_env import (
    BaseEnv,
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    ActionTuple,
    BehaviorName,
    AgentId,
    BehaviorMapping,
)
from mlagents_envs.timers import timed, hierarchical_timer
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityActionException,
    UnityTimeOutException,
    UnityCommunicatorStoppedException,
)

from mlagents_envs.communicator_objects.command_pb2 import STEP, RESET
from mlagents_envs.rpc_utils import behavior_spec_from_proto, steps_from_proto

from mlagents_envs.communicator_objects.unity_rl_input_pb2 import UnityRLInputProto
from mlagents_envs.communicator_objects.unity_rl_output_pb2 import UnityRLOutputProto
from mlagents_envs.communicator_objects.agent_action_pb2 import AgentActionProto
from mlagents_envs.communicator_objects.unity_output_pb2 import UnityOutputProto
from mlagents_envs.communicator_objects.capabilities_pb2 import UnityRLCapabilitiesProto
from mlagents_envs.communicator_objects.unity_rl_initialization_input_pb2 import (
    UnityRLInitializationInputProto,
)

from mlagents_envs.communicator_objects.unity_input_pb2 import UnityInputProto

from .rpc_communicator import RpcCommunicator
import signal

logger = get_logger(__name__)


class UnityEnvironment(BaseEnv):
    # Communication protocol version.
    # When connecting to C#, this must be compatible with Academy.k_ApiVersion.
    # We follow semantic versioning on the communication version, so existing
    # functionality will work as long the major versions match.
    # This should be changed whenever a change is made to the communication protocol.
    # Revision history:
    #  * 1.0.0 - initial version
    #  * 1.1.0 - support concatenated PNGs for compressed observations.
    #  * 1.2.0 - support compression mapping for stacked compressed observations.
    #  * 1.3.0 - support action spaces with both continuous and discrete actions.
    #  * 1.4.0 - support training analytics sent from python trainer to the editor.
    #  * 1.5.0 - support variable length observation training and multi-agent groups.
    API_VERSION = "1.5.0"

    # Default port that the editor listens on. If an environment executable
    # isn't specified, this port will be used.
    DEFAULT_EDITOR_PORT = 5004

    # Default base port for environments. Each environment will be offset from this
    # by it's worker_id.
    BASE_ENVIRONMENT_PORT = 5005

    # Command line argument used to pass the port to the executable environment.
    _PORT_COMMAND_LINE_ARG = "--mlagents-port"

    @staticmethod
    def _raise_version_exception(unity_com_ver: str) -> None:
        raise UnityEnvironmentException(
            f"The communication API version is not compatible between Unity and python. "
            f"Python API: {UnityEnvironment.API_VERSION}, Unity API: {unity_com_ver}.\n "
            f"Please find the versions that work best together from our release page.\n"
            "https://github.com/Unity-Technologies/ml-agents/releases"
        )

    @staticmethod
    def _check_communication_compatibility(
        unity_com_ver: str, python_api_version: str, unity_package_version: str
    ) -> bool:
        unity_communicator_version = StrictVersion(unity_com_ver)
        api_version = StrictVersion(python_api_version)
        if unity_communicator_version.version[0] == 0:
            if (
                unity_communicator_version.version[0] != api_version.version[0]
                or unity_communicator_version.version[1] != api_version.version[1]
            ):
                # Minor beta versions differ.
                return False
        elif unity_communicator_version.version[0] != api_version.version[0]:
            # Major versions mismatch.
            return False
        else:
            # Major versions match, so either:
            # 1) The versions are identical, in which case there's no compatibility issues
            # 2) The Unity version is newer, in which case we'll warn or fail on the Unity side if trying to use
            #    unsupported features
            # 3) The trainer version is newer, in which case new trainer features might be available but unused by C#
            # In any of the cases, there's no reason to warn about mismatch here.
            logger.info(
                f"Connected to Unity environment with package version {unity_package_version} "
                f"and communication version {unity_com_ver}"
            )
        return True

    @staticmethod
    def _get_capabilities_proto() -> UnityRLCapabilitiesProto:
        capabilities = UnityRLCapabilitiesProto()
        capabilities.baseRLCapabilities = True
        capabilities.concatenatedPngObservations = True
        capabilities.compressedChannelMapping = True
        capabilities.hybridActions = True
        capabilities.trainingAnalytics = True
        capabilities.variableLengthObservation = True
        capabilities.multiAgentGroups = True
        return capabilities

    @staticmethod
    def _warn_csharp_base_capabilities(
        caps: UnityRLCapabilitiesProto, unity_package_ver: str, python_package_ver: str
    ) -> None:
        if not caps.baseRLCapabilities:
            logger.warning(
                "WARNING: The Unity process is not running with the expected base Reinforcement Learning"
                " capabilities. Please be sure upgrade the Unity Package to a version that is compatible with this "
                "python package.\n"
                f"Python package version: {python_package_ver}, C# package version: {unity_package_ver}"
                f"Please find the versions that work best together from our release page.\n"
                "https://github.com/Unity-Technologies/ml-agents/releases"
            )

    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        additional_args: Optional[List[str]] = None,
        side_channels: Optional[List[SideChannel]] = None,
        log_folder: Optional[str] = None,
        num_areas: int = 1,
    ):
        """
        Starts a new unity environment and establishes a connection with the environment.
        Notice: Currently communication between Unity and Python takes place over an open socket without authentication.
        Ensure that the network where training takes place is secure.

        :string file_name: Name of Unity environment binary.
        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        If no environment is specified (i.e. file_name is None), the DEFAULT_EDITOR_PORT will be used.
        :int worker_id: Offset from base_port. Used for training multiple environments simultaneously.
        :bool no_graphics: Whether to run the Unity simulator in no-graphics mode
        :int timeout_wait: Time (in seconds) to wait for connection from environment.
        :list args: Addition Unity command line arguments
        :list side_channels: Additional side channel for no-rl communication with Unity
        :str log_folder: Optional folder to write the Unity Player log file into.  Requires absolute path.
        """
        atexit.register(self._close)
        self._additional_args = additional_args or []
        self._no_graphics = no_graphics
        # If base port is not specified, use BASE_ENVIRONMENT_PORT if we have
        # an environment, otherwise DEFAULT_EDITOR_PORT
        if base_port is None:
            base_port = (
                self.BASE_ENVIRONMENT_PORT if file_name else self.DEFAULT_EDITOR_PORT
            )
        self._port = base_port + worker_id
        self._buffer_size = 12000
        # If true, this means the environment was successfully loaded
        self._loaded = False
        # The process that is started. If None, no process was started
        self._process: Optional[subprocess.Popen] = None
        self._timeout_wait: int = timeout_wait
        self._communicator = self._get_communicator(worker_id, base_port, timeout_wait)
        self._worker_id = worker_id
        if side_channels is None:
            side_channels = []
        default_training_side_channel: Optional[
            DefaultTrainingAnalyticsSideChannel
        ] = None
        if DefaultTrainingAnalyticsSideChannel.CHANNEL_ID not in [
            _.channel_id for _ in side_channels
        ]:
            default_training_side_channel = DefaultTrainingAnalyticsSideChannel()
            side_channels.append(default_training_side_channel)
        self._side_channel_manager = SideChannelManager(side_channels)
        self._log_folder = log_folder
        self.academy_capabilities: UnityRLCapabilitiesProto = None  # type: ignore

        # If the environment name is None, a new environment will not be launched
        # and the communicator will directly try to connect to an existing unity environment.
        # If the worker-id is not 0 and the environment name is None, an error is thrown
        if file_name is None and worker_id != 0:
            raise UnityEnvironmentException(
                "If the environment name is None, "
                "the worker-id must be 0 in order to connect with the Editor."
            )
        if file_name is not None:
            try:
                self._process = env_utils.launch_executable(
                    file_name, self._executable_args()
                )
            except UnityEnvironmentException:
                self._close(0)
                raise
        else:
            logger.info(
                f"Listening on port {self._port}. "
                f"Start training by pressing the Play button in the Unity Editor."
            )
        self._loaded = True

        rl_init_parameters_in = UnityRLInitializationInputProto(
            seed=seed,
            communication_version=self.API_VERSION,
            package_version=mlagents_envs.__version__,
            capabilities=UnityEnvironment._get_capabilities_proto(),
            num_areas=num_areas,
        )
        try:
            aca_output = self._send_academy_parameters(rl_init_parameters_in)
            aca_params = aca_output.rl_initialization_output
        except UnityTimeOutException:
            self._close(0)
            raise

        if not UnityEnvironment._check_communication_compatibility(
            aca_params.communication_version,
            UnityEnvironment.API_VERSION,
            aca_params.package_version,
        ):
            self._close(0)
            UnityEnvironment._raise_version_exception(aca_params.communication_version)

        UnityEnvironment._warn_csharp_base_capabilities(
            aca_params.capabilities,
            aca_params.package_version,
            UnityEnvironment.API_VERSION,
        )

        self._env_state: Dict[str, Tuple[DecisionSteps, TerminalSteps]] = {}
        self._env_specs: Dict[str, BehaviorSpec] = {}
        self._env_actions: Dict[str, ActionTuple] = {}
        self._is_first_message = True
        self._update_behavior_specs(aca_output)
        self.academy_capabilities = aca_params.capabilities
        if default_training_side_channel is not None:
            default_training_side_channel.environment_initialized()

    @staticmethod
    def _get_communicator(worker_id, base_port, timeout_wait):
        return RpcCommunicator(worker_id, base_port, timeout_wait)

    def _executable_args(self) -> List[str]:
        args: List[str] = []
        if self._no_graphics:
            args += ["-nographics", "-batchmode"]
        args += [UnityEnvironment._PORT_COMMAND_LINE_ARG, str(self._port)]

        # If the logfile arg isn't already set in the env args,
        # try to set it to an output directory
        logfile_set = "-logfile" in (arg.lower() for arg in self._additional_args)
        if self._log_folder and not logfile_set:
            log_file_path = os.path.join(
                self._log_folder, f"Player-{self._worker_id}.log"
            )
            args += ["-logFile", log_file_path]
        # Add in arguments passed explicitly by the user.
        args += self._additional_args
        return args

    def _update_behavior_specs(self, output: UnityOutputProto) -> None:
        init_output = output.rl_initialization_output
        for brain_param in init_output.brain_parameters:
            # Each BrainParameter in the rl_initialization_output should have at least one AgentInfo
            # Get that agent, because we need some of its observations.
            agent_infos = output.rl_output.agentInfos[brain_param.brain_name]
            if agent_infos.value:
                agent = agent_infos.value[0]
                new_spec = behavior_spec_from_proto(brain_param, agent)
                self._env_specs[brain_param.brain_name] = new_spec
                logger.info(f"Connected new brain: {brain_param.brain_name}")

    def _update_state(self, output: UnityRLOutputProto) -> None:
        """
        Collects experience information from all external brains in environment at current step.
        """
        for brain_name in self._env_specs.keys():
            if brain_name in output.agentInfos:
                agent_info_list = output.agentInfos[brain_name].value
                self._env_state[brain_name] = steps_from_proto(
                    agent_info_list, self._env_specs[brain_name]
                )
            else:
                self._env_state[brain_name] = (
                    DecisionSteps.empty(self._env_specs[brain_name]),
                    TerminalSteps.empty(self._env_specs[brain_name]),
                )
        self._side_channel_manager.process_side_channel_message(output.side_channel)

    def reset(self) -> None:
        if self._loaded:
            outputs = self._communicator.exchange(
                self._generate_reset_input(), self._poll_process
            )
            if outputs is None:
                raise UnityCommunicatorStoppedException("Communicator has exited.")
            self._update_behavior_specs(outputs)
            rl_output = outputs.rl_output
            self._update_state(rl_output)
            self._is_first_message = False
            self._env_actions.clear()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    @timed
    def step(self) -> None:
        if self._is_first_message:
            return self.reset()
        if not self._loaded:
            raise UnityEnvironmentException("No Unity environment is loaded.")
        # fill the blanks for missing actions
        for group_name in self._env_specs:
            if group_name not in self._env_actions:
                n_agents = 0
                if group_name in self._env_state:
                    n_agents = len(self._env_state[group_name][0])
                self._env_actions[group_name] = self._env_specs[
                    group_name
                ].action_spec.empty_action(n_agents)
        step_input = self._generate_step_input(self._env_actions)
        with hierarchical_timer("communicator.exchange"):
            outputs = self._communicator.exchange(step_input, self._poll_process)
        if outputs is None:
            raise UnityCommunicatorStoppedException("Communicator has exited.")
        self._update_behavior_specs(outputs)
        rl_output = outputs.rl_output
        self._update_state(rl_output)
        self._env_actions.clear()

    @property
    def behavior_specs(self) -> MappingType[str, BehaviorSpec]:
        return BehaviorMapping(self._env_specs)

    def _assert_behavior_exists(self, behavior_name: str) -> None:
        if behavior_name not in self._env_specs:
            raise UnityActionException(
                f"The group {behavior_name} does not correspond to an existing "
                f"agent group in the environment"
            )

    def set_actions(self, behavior_name: BehaviorName, action: ActionTuple) -> None:
        self._assert_behavior_exists(behavior_name)
        if behavior_name not in self._env_state:
            return
        action_spec = self._env_specs[behavior_name].action_spec
        num_agents = len(self._env_state[behavior_name][0])
        action = action_spec._validate_action(action, num_agents, behavior_name)
        self._env_actions[behavior_name] = action

    def set_action_for_agent(
        self, behavior_name: BehaviorName, agent_id: AgentId, action: ActionTuple
    ) -> None:
        self._assert_behavior_exists(behavior_name)
        if behavior_name not in self._env_state:
            return
        action_spec = self._env_specs[behavior_name].action_spec
        action = action_spec._validate_action(action, 1, behavior_name)
        if behavior_name not in self._env_actions:
            num_agents = len(self._env_state[behavior_name][0])
            self._env_actions[behavior_name] = action_spec.empty_action(num_agents)
        try:
            index = np.where(self._env_state[behavior_name][0].agent_id == agent_id)[0][
                0
            ]
        except IndexError as ie:
            raise IndexError(
                "agent_id {} is did not request a decision at the previous step".format(
                    agent_id
                )
            ) from ie
        if action_spec.continuous_size > 0:
            self._env_actions[behavior_name].continuous[index] = action.continuous[0, :]
        if action_spec.discrete_size > 0:
            self._env_actions[behavior_name].discrete[index] = action.discrete[0, :]

    def get_steps(
        self, behavior_name: BehaviorName
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        self._assert_behavior_exists(behavior_name)
        return self._env_state[behavior_name]

    def _poll_process(self) -> None:
        """
        Check the status of the subprocess. If it has exited, raise a UnityEnvironmentException
        :return: None
        """
        if not self._process:
            return
        poll_res = self._process.poll()
        if poll_res is not None:
            exc_msg = self._returncode_to_env_message(self._process.returncode)
            raise UnityEnvironmentException(exc_msg)

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._loaded:
            self._close()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    def _close(self, timeout: Optional[int] = None) -> None:
        """
        Close the communicator and environment subprocess (if necessary).

        :int timeout: [Optional] Number of seconds to wait for the environment to shut down before
            force-killing it.  Defaults to `self.timeout_wait`.
        """
        if timeout is None:
            timeout = self._timeout_wait
        self._loaded = False
        self._communicator.close()
        if self._process is not None:
            # Wait a bit for the process to shutdown, but kill it if it takes too long
            try:
                self._process.wait(timeout=timeout)
                logger.debug(self._returncode_to_env_message(self._process.returncode))
            except subprocess.TimeoutExpired:
                logger.warning("Environment timed out shutting down. Killing...")
                self._process.kill()
            # Set to None so we don't try to close multiple times.
            self._process = None

    @timed
    def _generate_step_input(
        self, vector_action: Dict[str, ActionTuple]
    ) -> UnityInputProto:
        rl_in = UnityRLInputProto()
        for b in vector_action:
            n_agents = len(self._env_state[b][0])
            if n_agents == 0:
                continue
            for i in range(n_agents):
                action = AgentActionProto()
                if vector_action[b].continuous is not None:
                    action.vector_actions_deprecated.extend(
                        vector_action[b].continuous[i]
                    )
                    action.continuous_actions.extend(vector_action[b].continuous[i])
                if vector_action[b].discrete is not None:
                    action.vector_actions_deprecated.extend(
                        vector_action[b].discrete[i]
                    )
                    action.discrete_actions.extend(vector_action[b].discrete[i])
                rl_in.agent_actions[b].value.extend([action])
                rl_in.command = STEP
        rl_in.side_channel = bytes(
            self._side_channel_manager.generate_side_channel_messages()
        )
        return self._wrap_unity_input(rl_in)

    def _generate_reset_input(self) -> UnityInputProto:
        rl_in = UnityRLInputProto()
        rl_in.command = RESET
        rl_in.side_channel = bytes(
            self._side_channel_manager.generate_side_channel_messages()
        )
        return self._wrap_unity_input(rl_in)

    def _send_academy_parameters(
        self, init_parameters: UnityRLInitializationInputProto
    ) -> UnityOutputProto:
        inputs = UnityInputProto()
        inputs.rl_initialization_input.CopyFrom(init_parameters)
        return self._communicator.initialize(inputs, self._poll_process)

    @staticmethod
    def _wrap_unity_input(rl_input: UnityRLInputProto) -> UnityInputProto:
        result = UnityInputProto()
        result.rl_input.CopyFrom(rl_input)
        return result

    @staticmethod
    def _returncode_to_signal_name(returncode: int) -> Optional[str]:
        """
        Try to convert return codes into their corresponding signal name.
        E.g. returncode_to_signal_name(-2) -> "SIGINT"
        """
        try:
            # A negative value -N indicates that the child was terminated by signal N (POSIX only).
            s = signal.Signals(-returncode)
            return s.name
        except Exception:
            # Should generally be a ValueError, but catch everything just in case.
            return None

    @staticmethod
    def _returncode_to_env_message(returncode: int) -> str:
        signal_name = UnityEnvironment._returncode_to_signal_name(returncode)
        signal_name = f" ({signal_name})" if signal_name else ""
        return f"Environment shut down with return code {returncode}{signal_name}."
