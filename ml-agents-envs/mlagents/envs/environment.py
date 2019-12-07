import atexit
import glob
import logging
import numpy as np
import os
import subprocess
from typing import Dict, List, Optional, Any

from mlagents.envs.side_channel.side_channel import SideChannel
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.timers import timed, hierarchical_timer
from .brain import AllBrainInfo, BrainInfo, BrainParameters
from .exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityActionException,
    UnityTimeOutException,
)

from mlagents.envs.communicator_objects.unity_rl_input_pb2 import UnityRLInputProto
from mlagents.envs.communicator_objects.unity_rl_output_pb2 import UnityRLOutputProto
from mlagents.envs.communicator_objects.agent_action_pb2 import AgentActionProto
from mlagents.envs.communicator_objects.unity_output_pb2 import UnityOutputProto
from mlagents.envs.communicator_objects.unity_rl_initialization_input_pb2 import (
    UnityRLInitializationInputProto,
)

from mlagents.envs.communicator_objects.unity_input_pb2 import UnityInputProto

from .rpc_communicator import RpcCommunicator
from sys import platform
import signal
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlagents.envs")


class UnityEnvironment(BaseUnityEnvironment):
    SCALAR_ACTION_TYPES = (int, np.int32, np.int64, float, np.float32, np.float64)
    SINGLE_BRAIN_ACTION_TYPES = SCALAR_ACTION_TYPES + (list, np.ndarray)
    API_VERSION = "API-12"

    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: int = 5005,
        seed: int = 0,
        docker_training: bool = False,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        args: Optional[List[str]] = None,
        side_channels: Optional[List[SideChannel]] = None,
    ):
        """
        Starts a new unity environment and establishes a connection with the environment.
        Notice: Currently communication between Unity and Python takes place over an open socket without authentication.
        Ensure that the network where training takes place is secure.

        :string file_name: Name of Unity environment binary.
        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        :bool docker_training: Informs this class whether the process is being run within a container.
        :bool no_graphics: Whether to run the Unity simulator in no-graphics mode
        :int timeout_wait: Time (in seconds) to wait for connection from environment.
        :bool train_mode: Whether to run in training mode, speeding up the simulation, by default.
        :list args: Addition Unity command line arguments
        :list side_channels: Additional side channel for no-rl communication with Unity
        """
        args = args or []
        atexit.register(self._close)
        self.port = base_port + worker_id
        self._buffer_size = 12000
        self._version_ = UnityEnvironment.API_VERSION
        # If true, this means the environment was successfully loaded
        self._loaded = False
        # The process that is started. If None, no process was started
        self.proc1 = None
        self.timeout_wait: int = timeout_wait
        self.communicator = self.get_communicator(worker_id, base_port, timeout_wait)
        self.worker_id = worker_id
        self.side_channels: Dict[int, SideChannel] = {}
        if side_channels is not None:
            for _sc in side_channels:
                if _sc.channel_type in self.side_channels:
                    raise UnityEnvironmentException(
                        "There cannot be two side channels with the same channel type {0}.".format(
                            _sc.channel_type
                        )
                    )
                self.side_channels[_sc.channel_type] = _sc

        # If the environment name is None, a new environment will not be launched
        # and the communicator will directly try to connect to an existing unity environment.
        # If the worker-id is not 0 and the environment name is None, an error is thrown
        if file_name is None and worker_id != 0:
            raise UnityEnvironmentException(
                "If the environment name is None, "
                "the worker-id must be 0 in order to connect with the Editor."
            )
        if file_name is not None:
            self.executable_launcher(file_name, docker_training, no_graphics, args)
        else:
            logger.info(
                f"Listening on port {self.port}. "
                f"Start training by pressing the Play button in the Unity Editor."
            )
        self._loaded = True

        rl_init_parameters_in = UnityRLInitializationInputProto(seed=seed)
        try:
            aca_output = self.send_academy_parameters(rl_init_parameters_in)
            aca_params = aca_output.rl_initialization_output
        except UnityTimeOutException:
            self._close()
            raise
        # TODO : think of a better way to expose the academyParameters
        self._unity_version = aca_params.version
        if self._unity_version != self._version_:
            self._close()
            raise UnityEnvironmentException(
                f"The API number is not compatible between Unity and python. "
                f"Python API: {self._version_}, Unity API: {self._unity_version}.\n"
                f"Please go to https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release"
                f"to download the latest version of ML-Agents."
            )
        self._n_agents: Dict[str, int] = {}
        self._is_first_message = True
        self._academy_name = aca_params.name
        self._log_path = aca_params.log_path
        self._brains: Dict[str, BrainParameters] = {}
        self._external_brain_names: List[str] = []
        self._num_external_brains = 0
        self._update_brain_parameters(aca_output)
        logger.info(
            "\n'{0}' started successfully!\n{1}".format(self._academy_name, str(self))
        )

    @property
    def logfile_path(self):
        return self._log_path

    @property
    def brains(self):
        return self._brains

    @property
    def academy_name(self):
        return self._academy_name

    @property
    def number_external_brains(self):
        return self._num_external_brains

    @property
    def external_brain_names(self):
        return self._external_brain_names

    @staticmethod
    def get_communicator(worker_id, base_port, timeout_wait):
        return RpcCommunicator(worker_id, base_port, timeout_wait)

    @property
    def external_brains(self):
        external_brains = {}
        for brain_name in self.external_brain_names:
            external_brains[brain_name] = self.brains[brain_name]
        return external_brains

    def executable_launcher(self, file_name, docker_training, no_graphics, args):
        cwd = os.getcwd()
        file_name = (
            file_name.strip()
            .replace(".app", "")
            .replace(".exe", "")
            .replace(".x86_64", "")
            .replace(".x86", "")
        )
        true_filename = os.path.basename(os.path.normpath(file_name))
        logger.debug("The true file name is {}".format(true_filename))
        launch_string = None
        if platform == "linux" or platform == "linux2":
            candidates = glob.glob(os.path.join(cwd, file_name) + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + ".x86")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86")
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform == "darwin":
            candidates = glob.glob(
                os.path.join(
                    cwd, file_name + ".app", "Contents", "MacOS", true_filename
                )
            )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", true_filename)
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(cwd, file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + ".app", "Contents", "MacOS", "*")
                )
            if len(candidates) > 0:
                launch_string = candidates[0]
        elif platform == "win32":
            candidates = glob.glob(os.path.join(cwd, file_name + ".exe"))
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".exe")
            if len(candidates) > 0:
                launch_string = candidates[0]
        if launch_string is None:
            self._close()
            raise UnityEnvironmentException(
                "Couldn't launch the {0} environment. "
                "Provided filename does not match any environments.".format(
                    true_filename
                )
            )
        else:
            logger.debug("This is the launch string {}".format(launch_string))
            # Launch Unity environment
            if not docker_training:
                subprocess_args = [launch_string]
                if no_graphics:
                    subprocess_args += ["-nographics", "-batchmode"]
                subprocess_args += ["--port", str(self.port)]
                subprocess_args += args
                try:
                    self.proc1 = subprocess.Popen(
                        subprocess_args,
                        # start_new_session=True means that signals to the parent python process
                        # (e.g. SIGINT from keyboard interrupt) will not be sent to the new process on POSIX platforms.
                        # This is generally good since we want the environment to have a chance to shutdown,
                        # but may be undesirable in come cases; if so, we'll add a command-line toggle.
                        # Note that on Windows, the CTRL_C signal will still be sent.
                        start_new_session=True,
                    )
                except PermissionError as perm:
                    # This is likely due to missing read or execute permissions on file.
                    raise UnityEnvironmentException(
                        f"Error when trying to launch environment - make sure "
                        f"permissions are set correctly. For example "
                        f'"chmod -R 755 {launch_string}"'
                    ) from perm

            else:
                # Comments for future maintenance:
                #     xvfb-run is a wrapper around Xvfb, a virtual xserver where all
                #     rendering is done to virtual memory. It automatically creates a
                #     new virtual server automatically picking a server number `auto-servernum`.
                #     The server is passed the arguments using `server-args`, we are telling
                #     Xvfb to create Screen number 0 with width 640, height 480 and depth 24 bits.
                #     Note that 640 X 480 are the default width and height. The main reason for
                #     us to add this is because we'd like to change the depth from the default
                #     of 8 bits to 24.
                #     Unfortunately, this means that we will need to pass the arguments through
                #     a shell which is why we set `shell=True`. Now, this adds its own
                #     complications. E.g SIGINT can bounce off the shell and not get propagated
                #     to the child processes. This is why we add `exec`, so that the shell gets
                #     launched, the arguments are passed to `xvfb-run`. `exec` replaces the shell
                #     we created with `xvfb`.
                #
                docker_ls = (
                    "exec xvfb-run --auto-servernum"
                    " --server-args='-screen 0 640x480x24'"
                    " {0} --port {1}"
                ).format(launch_string, str(self.port))
                self.proc1 = subprocess.Popen(
                    docker_ls,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                )

    def __str__(self):
        return """Unity Academy name: {0}""".format(self._academy_name)

    def reset(self) -> AllBrainInfo:
        """
        Sends a signal to reset the unity environment.
        :return: AllBrainInfo  : A data structure corresponding to the initial reset state of the environment.
        """
        if self._loaded:
            outputs = self.communicator.exchange(self._generate_reset_input())
            if outputs is None:
                raise UnityCommunicationException("Communicator has stopped.")
            self._update_brain_parameters(outputs)
            rl_output = outputs.rl_output
            s = self._get_state(rl_output)
            for _b in self._external_brain_names:
                self._n_agents[_b] = len(s[_b].agents)
            self._is_first_message = False
            return s
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    @timed
    def step(
        self,
        vector_action: Dict[str, np.ndarray] = None,
        value: Optional[Dict[str, np.ndarray]] = None,
    ) -> AllBrainInfo:
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly,
        and returns observation, state, and reward information to the agent.
        :param value: Value estimates provided by agents.
        :param vector_action: Agent's vector action. Can be a scalar or vector of int/floats.
        :param memory: Vector corresponding to memory used for recurrent policies.
        :return: AllBrainInfo  : A Data structure corresponding to the new state of the environment.
        """
        if self._is_first_message:
            return self.reset()
        vector_action = {} if vector_action is None else vector_action
        value = {} if value is None else value

        # Check that environment is loaded, and episode is currently running.
        if not self._loaded:
            raise UnityEnvironmentException("No Unity environment is loaded.")
        else:
            if isinstance(vector_action, self.SINGLE_BRAIN_ACTION_TYPES):
                if self._num_external_brains == 1:
                    vector_action = {self._external_brain_names[0]: vector_action}
                elif self._num_external_brains > 1:
                    raise UnityActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names a keys, "
                        "and vector_actions as values".format(self._num_external_brains)
                    )
                else:
                    raise UnityActionException(
                        "There are no external brains in the environment, "
                        "step cannot take a vector_action input"
                    )

            if isinstance(value, self.SINGLE_BRAIN_ACTION_TYPES):
                if self._num_external_brains == 1:
                    value = {self._external_brain_names[0]: value}
                elif self._num_external_brains > 1:
                    raise UnityActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names as keys "
                        "and state/action value estimates as values".format(
                            self._num_external_brains
                        )
                    )
                else:
                    raise UnityActionException(
                        "There are no external brains in the environment, "
                        "step cannot take a value input"
                    )

            for brain_name in list(vector_action.keys()):
                if brain_name not in self._external_brain_names:
                    raise UnityActionException(
                        "The name {0} does not correspond to an external brain "
                        "in the environment".format(brain_name)
                    )

            for brain_name in self._external_brain_names:
                n_agent = self._n_agents[brain_name]
                if brain_name not in vector_action:
                    if self._brains[brain_name].vector_action_space_type == "discrete":
                        vector_action[brain_name] = (
                            [0.0]
                            * n_agent
                            * len(self._brains[brain_name].vector_action_space_size)
                        )
                    else:
                        vector_action[brain_name] = (
                            [0.0]
                            * n_agent
                            * self._brains[brain_name].vector_action_space_size[0]
                        )
                else:
                    vector_action[brain_name] = self._flatten(vector_action[brain_name])

                discrete_check = (
                    self._brains[brain_name].vector_action_space_type == "discrete"
                )

                expected_discrete_size = n_agent * len(
                    self._brains[brain_name].vector_action_space_size
                )

                continuous_check = (
                    self._brains[brain_name].vector_action_space_type == "continuous"
                )

                expected_continuous_size = (
                    self._brains[brain_name].vector_action_space_size[0] * n_agent
                )

                if not (
                    (
                        discrete_check
                        and len(vector_action[brain_name]) == expected_discrete_size
                    )
                    or (
                        continuous_check
                        and len(vector_action[brain_name]) == expected_continuous_size
                    )
                ):
                    raise UnityActionException(
                        "There was a mismatch between the provided action and "
                        "the environment's expectation: "
                        "The brain {0} expected {1} {2} action(s), but was provided: {3}".format(
                            brain_name,
                            str(expected_discrete_size)
                            if discrete_check
                            else str(expected_continuous_size),
                            self._brains[brain_name].vector_action_space_type,
                            str(vector_action[brain_name]),
                        )
                    )

            step_input = self._generate_step_input(vector_action, value)
            with hierarchical_timer("communicator.exchange"):
                outputs = self.communicator.exchange(step_input)
            if outputs is None:
                raise UnityCommunicationException("Communicator has stopped.")
            self._update_brain_parameters(outputs)
            rl_output = outputs.rl_output
            state = self._get_state(rl_output)
            for _b in self._external_brain_names:
                self._n_agents[_b] = len(state[_b].agents)
            return state

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._loaded:
            self._close()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    def _close(self):
        self._loaded = False
        self.communicator.close()
        if self.proc1 is not None:
            # Wait a bit for the process to shutdown, but kill it if it takes too long
            try:
                self.proc1.wait(timeout=self.timeout_wait)
                signal_name = self.returncode_to_signal_name(self.proc1.returncode)
                signal_name = f" ({signal_name})" if signal_name else ""
                return_info = f"Environment shut down with return code {self.proc1.returncode}{signal_name}."
                logger.info(return_info)
            except subprocess.TimeoutExpired:
                logger.info("Environment timed out shutting down. Killing...")
                self.proc1.kill()
            # Set to None so we don't try to close multiple times.
            self.proc1 = None

    @classmethod
    def _flatten(cls, arr: Any) -> List[float]:
        """
        Converts arrays to list.
        :param arr: numpy vector.
        :return: flattened list.
        """
        if isinstance(arr, cls.SCALAR_ACTION_TYPES):
            arr = [float(arr)]
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if len(arr) == 0:
            return arr
        if isinstance(arr[0], np.ndarray):
            # pylint: disable=no-member
            arr = [item for sublist in arr for item in sublist.tolist()]
        if isinstance(arr[0], list):
            # pylint: disable=not-an-iterable
            arr = [item for sublist in arr for item in sublist]
        arr = [float(x) for x in arr]
        return arr

    def _get_state(self, output: UnityRLOutputProto) -> AllBrainInfo:
        """
        Collects experience information from all external brains in environment at current step.
        :return: a dictionary of BrainInfo objects.
        """
        _data = {}
        for brain_name in output.agentInfos:
            agent_info_list = output.agentInfos[brain_name].value
            _data[brain_name] = BrainInfo.from_agent_proto(
                self.worker_id, agent_info_list, self.brains[brain_name]
            )
        self._parse_side_channel_message(self.side_channels, output.side_channel)
        return _data

    @staticmethod
    def _parse_side_channel_message(
        side_channels: Dict[int, SideChannel], data: bytearray
    ) -> None:
        offset = 0
        while offset < len(data):
            try:
                channel_type, message_len = struct.unpack_from("<ii", data, offset)
                offset = offset + 8
                message_data = data[offset : offset + message_len]
                offset = offset + message_len
            except Exception:
                raise UnityEnvironmentException(
                    "There was a problem reading a message in a SideChannel. "
                    "Please make sure the version of MLAgents in Unity is "
                    "compatible with the Python version."
                )
            if len(message_data) != message_len:
                raise UnityEnvironmentException(
                    "The message received by the side channel {0} was "
                    "unexpectedly short. Make sure your Unity Environment "
                    "sending side channel data properly.".format(channel_type)
                )
            if channel_type in side_channels:
                side_channels[channel_type].on_message_received(message_data)
            else:
                logger.warning(
                    "Unknown side channel data received. Channel type "
                    ": {0}.".format(channel_type)
                )

    @staticmethod
    def _generate_side_channel_data(side_channels: Dict[int, SideChannel]) -> bytearray:
        result = bytearray()
        for channel_type, channel in side_channels.items():
            for message in channel.message_queue:
                result += struct.pack("<ii", channel_type, len(message))
                result += message
            channel.message_queue = []
        return result

    def _update_brain_parameters(self, output: UnityOutputProto) -> None:
        init_output = output.rl_initialization_output

        for brain_param in init_output.brain_parameters:
            # Each BrainParameter in the rl_initialization_output should have at least one AgentInfo
            # Get that agent, because we need some of its observations.
            agent_infos = output.rl_output.agentInfos[brain_param.brain_name]
            if agent_infos.value:
                agent = agent_infos.value[0]
                new_brain = BrainParameters.from_proto(brain_param, agent)
                self._brains[brain_param.brain_name] = new_brain
                logger.info(f"Connected new brain:\n{new_brain}")
        self._external_brain_names = list(self._brains.keys())
        self._num_external_brains = len(self._external_brain_names)

    @timed
    def _generate_step_input(
        self, vector_action: Dict[str, np.ndarray], value: Dict[str, np.ndarray]
    ) -> UnityInputProto:
        rl_in = UnityRLInputProto()
        for b in vector_action:
            n_agents = self._n_agents[b]
            if n_agents == 0:
                continue
            _a_s = len(vector_action[b]) // n_agents
            for i in range(n_agents):
                action = AgentActionProto(
                    vector_actions=vector_action[b][i * _a_s : (i + 1) * _a_s]
                )
                if b in value:
                    if value[b] is not None:
                        action.value = float(value[b][i])
                rl_in.agent_actions[b].value.extend([action])
                rl_in.command = 0
        rl_in.side_channel = bytes(self._generate_side_channel_data(self.side_channels))
        return self.wrap_unity_input(rl_in)

    def _generate_reset_input(self) -> UnityInputProto:
        rl_in = UnityRLInputProto()
        rl_in.command = 1
        rl_in.side_channel = bytes(self._generate_side_channel_data(self.side_channels))
        return self.wrap_unity_input(rl_in)

    def send_academy_parameters(
        self, init_parameters: UnityRLInitializationInputProto
    ) -> UnityOutputProto:
        inputs = UnityInputProto()
        inputs.rl_initialization_input.CopyFrom(init_parameters)
        return self.communicator.initialize(inputs)

    @staticmethod
    def wrap_unity_input(rl_input: UnityRLInputProto) -> UnityInputProto:
        result = UnityInputProto()
        result.rl_input.CopyFrom(rl_input)
        return result

    @staticmethod
    def returncode_to_signal_name(returncode: int) -> Optional[str]:
        """
        Try to convert return codes into their corresponding signal name.
        E.g. returncode_to_signal_name(-2) -> "SIGINT"
        """
        try:
            # A negative value -N indicates that the child was terminated by signal N (POSIX only).
            s = signal.Signals(-returncode)  # pylint: disable=no-member
            return s.name
        except Exception:
            # Should generally be a ValueError, but catch everything just in case.
            return None
