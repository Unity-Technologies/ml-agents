import atexit
import io
import glob
import json
import logging
import numpy as np
import os
import socket
import subprocess
import struct

from .brain import BrainInfo, BrainParameters
from .exception import UnityEnvironmentException, UnityActionException, UnityTimeOutException
from .curriculum import Curriculum

from PIL import Image
from sys import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unityagents")


class UnityEnvironment(object):
    def __init__(self, file_name, worker_id=0,
                 base_port=5005, curriculum=None):
        """
        Starts a new unity environment and establishes a connection with the environment.
        Notice: Currently communication between Unity and Python takes place over an open socket without authentication.
        Ensure that the network where training takes place is secure.

        :string file_name: Name of Unity environment binary.
        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """

        atexit.register(self.close)
        self.port = base_port + worker_id
        self._buffer_size = 12000
        self._python_api = "API-2"
        self._loaded = False
        self._open_socket = False

        try:
            # Establish communication socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("localhost", self.port))
            self._open_socket = True
        except socket.error:
            self._open_socket = True
            self.close()
            raise socket.error("Couldn't launch new environment because worker number {} is still in use. "
                               "You may need to manually close a previously opened environment "
                               "or use a different worker number.".format(str(worker_id)))

        cwd = os.getcwd()
        file_name = (file_name.strip()
                     .replace('.app', '').replace('.exe', '').replace('.x86_64', '').replace('.x86', ''))
        true_filename = os.path.basename(os.path.normpath(file_name))
        launch_string = None
        if platform == "linux" or platform == "linux2":
            candidates = glob.glob(os.path.join(cwd, file_name) + '.x86_64')
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + '.x86')
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.x86_64')
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.x86')
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform == 'darwin':
            candidates = glob.glob(os.path.join(cwd, file_name + '.app', 'Contents', 'MacOS', true_filename))
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(file_name + '.app', 'Contents', 'MacOS', true_filename))
            if len(candidates) > 0:
                launch_string = candidates[0]
        elif platform == 'win32':
            candidates = glob.glob(os.path.join(cwd, file_name + '.exe'))
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.exe')
            if len(candidates) > 0:
                launch_string = candidates[0]
        if launch_string is None:
            self.close()
            raise UnityEnvironmentException("Couldn't launch the {0} environment. "
                                            "Provided filename does not match any environments."
                                            .format(true_filename))
        else:
            # Launch Unity environment
            proc1 = subprocess.Popen(
                [launch_string,
                 '--port', str(self.port)])

        self._socket.settimeout(30)
        try:
            try:
                self._socket.listen(1)
                self._conn, _ = self._socket.accept()
                self._conn.settimeout(30)
                p = self._conn.recv(self._buffer_size).decode('utf-8')
                p = json.loads(p)
            except socket.timeout as e:
                raise UnityTimeOutException(
                    "The Unity environment took too long to respond. Make sure {} does not need user interaction to "
                    "launch and that the Academy and the external Brain(s) are attached to objects in the Scene."
                    .format(str(file_name)))

            if "apiNumber" not in p:
                self._unity_api = "API-1"
            else:
                self._unity_api = p["apiNumber"]
            if self._unity_api != self._python_api:
                raise UnityEnvironmentException(
                    "The API number is not compatible between Unity and python. Python API : {0}, Unity API : "
                    "{1}.\nPlease go to https://github.com/Unity-Technologies/ml-agents to download the latest version "
                    "of ML-Agents.".format(self._python_api, self._unity_api))

            self._data = {}
            self._global_done = None
            self._academy_name = p["AcademyName"]
            self._log_path = p["logPath"]
            self._brains = {}
            self._brain_names = p["brainNames"]
            self._external_brain_names = p["externalBrainNames"]
            self._external_brain_names = [] if self._external_brain_names is None else self._external_brain_names
            self._num_brains = len(self._brain_names)
            self._num_external_brains = len(self._external_brain_names)
            self._resetParameters = p["resetParameters"]
            self._curriculum = Curriculum(curriculum, self._resetParameters)
            for i in range(self._num_brains):
                self._brains[self._brain_names[i]] = BrainParameters(self._brain_names[i], p["brainParameters"][i])
            self._loaded = True
            logger.info("\n'{}' started successfully!".format(self._academy_name))
            if (self._num_external_brains == 0):
                logger.warning(" No External Brains found in the Unity Environment. "
                               "You will not be able to pass actions to your agent(s).")
        except UnityEnvironmentException:
            proc1.kill()
            self.close()
            raise

    @property
    def logfile_path(self):
        return self._log_path

    @property
    def brains(self):
        return self._brains

    @property
    def global_done(self):
        return self._global_done

    @property
    def academy_name(self):
        return self._academy_name

    @property
    def number_brains(self):
        return self._num_brains

    @property
    def number_external_brains(self):
        return self._num_external_brains

    @property
    def brain_names(self):
        return self._brain_names

    @property
    def external_brain_names(self):
        return self._external_brain_names

    @staticmethod
    def _process_pixels(image_bytes=None, bw=False):
        """
        Converts bytearray observation image into numpy array, resizes it, and optionally converts it to greyscale
        :param image_bytes: input bytearray corresponding to image
        :return: processed numpy array of observation from environment
        """
        s = bytearray(image_bytes)
        image = Image.open(io.BytesIO(s))
        s = np.array(image) / 255.0
        if bw:
            s = np.mean(s, axis=2)
            s = np.reshape(s, [s.shape[0], s.shape[1], 1])
        return s

    def __str__(self):
        return '''Unity Academy name: {0}
        Number of brains: {1}
        Reset Parameters :\n\t\t{2}'''.format(self._academy_name, str(self._num_brains),
                                              "\n\t\t".join([str(k) + " -> " + str(self._resetParameters[k])
                                                             for k in self._resetParameters])) + '\n' + \
               '\n'.join([str(self._brains[b]) for b in self._brains])


    def _recv_bytes(self):
        try:
            s = self._conn.recv(self._buffer_size)
            message_length = struct.unpack("I", bytearray(s[:4]))[0]
            s = s[4:]
            while len(s) != message_length:
                s += self._conn.recv(self._buffer_size)
        except socket.timeout as e:
            raise UnityTimeOutException("The environment took too long to respond.", self._log_path)
        return s

    def _get_state_image(self, bw):
        """
        Receives observation from socket, and confirms.
        :param bw:
        :return:
        """
        s = self._recv_bytes()
        s = self._process_pixels(image_bytes=s, bw=bw)
        self._conn.send(b"RECEIVED")
        return s

    def _get_state_dict(self):
        """
        Receives dictionary of state information from socket, and confirms.
        :return:
        """
        state = self._recv_bytes().decode('utf-8')
        self._conn.send(b"RECEIVED")
        state_dict = json.loads(state)
        return state_dict

    def reset(self, train_mode=True, config=None, progress=None):
        """
        Sends a signal to reset the unity environment.
        :return: A Data structure corresponding to the initial reset state of the environment.
        """
        old_lesson = self._curriculum.get_lesson_number()
        if config is None:
            config = self._curriculum.get_lesson(progress)
            if old_lesson != self._curriculum.get_lesson_number():
                logger.info("\nLesson changed. Now in Lesson {0} : \t{1}"
                            .format(self._curriculum.get_lesson_number(),
                                    ', '.join([str(x) + ' -> ' + str(config[x]) for x in config])))
        elif config != {}:
            logger.info("\nAcademy Reset with parameters : \t{0}"
                        .format(', '.join([str(x) + ' -> ' + str(config[x]) for x in config])))
        for k in config:
            if (k in self._resetParameters) and (isinstance(config[k], (int, float))):
                self._resetParameters[k] = config[k]
            elif not isinstance(config[k], (int, float)):
                raise UnityEnvironmentException(
                    "The value for parameter '{0}'' must be an Integer or a Float.".format(k))
            else:
                raise UnityEnvironmentException("The parameter '{0}' is not a valid parameter.".format(k))

        if self._loaded:
            self._conn.send(b"RESET")
            try:
                self._conn.recv(self._buffer_size)
            except socket.timeout as e:
                raise UnityTimeOutException("The environment took too long to respond.", self._log_path)
            self._conn.send(json.dumps({"train_model": train_mode, "parameters": config}).encode('utf-8'))
            return self._get_state()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    def _get_state(self):
        """
        Collects experience information from all external brains in environment at current step.
        :return: a dictionary BrainInfo objects.
        """
        self._data = {}
        for index in range(self._num_brains):
            state_dict = self._get_state_dict()
            b = state_dict["brain_name"]
            n_agent = len(state_dict["agents"])
            try:
                if self._brains[b].state_space_type == "continuous":
                    states = np.array(state_dict["states"]).reshape((n_agent, self._brains[b].state_space_size))
                else:
                    states = np.array(state_dict["states"]).reshape((n_agent, 1))
            except UnityActionException:
                raise UnityActionException("Brain {0} has an invalid state. "
                                           "Expecting {1} {2} state but received {3}."
                                           .format(b, n_agent if self._brains[b].state_space_type == "discrete"
                else str(self._brains[b].state_space_size * n_agent),
                                                   self._brains[b].state_space_type,
                                                   len(state_dict["states"])))
            memories = np.array(state_dict["memories"]).reshape((n_agent, self._brains[b].memory_space_size))
            rewards = state_dict["rewards"]
            dones = state_dict["dones"]
            agents = state_dict["agents"]
            # actions = state_dict["actions"]
            if n_agent > 0:
                actions = np.array(state_dict["actions"]).reshape((n_agent, -1))
            else:
                actions = np.array([])

            observations = []
            for o in range(self._brains[b].number_observations):
                obs_n = []
                for a in range(n_agent):
                    obs_n.append(self._get_state_image(self._brains[b].camera_resolutions[o]['blackAndWhite']))

                observations.append(np.array(obs_n))

            self._data[b] = BrainInfo(observations, states, memories, rewards, agents, dones, actions)

        try:
            self._global_done = self._conn.recv(self._buffer_size).decode('utf-8') == 'True'
        except socket.timeout as e:
            raise UnityTimeOutException("The environment took too long to respond.", self._log_path)

        return self._data

    def _send_action(self, action, memory, value):
        """
        Send dictionary of actions, memories, and value estimates over socket.
        :param action: a dictionary of lists of actions.
        :param memory: a dictionary of lists of of memories.
        :param value: a dictionary of lists of of value estimates.
        """
        try:
            self._conn.recv(self._buffer_size)
        except socket.timeout as e:
            raise UnityTimeOutException("The environment took too long to respond.", self._log_path)
        action_message = {"action": action, "memory": memory, "value": value}
        self._conn.send(json.dumps(action_message).encode('utf-8'))

    @staticmethod
    def _flatten(arr):
        """
        Converts dictionary of arrays to list for transmission over socket.
        :param arr: numpy vector.
        :return: flattened list.
        """
        if isinstance(arr, (int, np.int_, float, np.float_)):
            arr = [float(arr)]
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if len(arr) == 0:
            return arr
        if isinstance(arr[0], np.ndarray):
            arr = [item for sublist in arr for item in sublist.tolist()]
        if isinstance(arr[0], list):
            arr = [item for sublist in arr for item in sublist]
        arr = [float(x) for x in arr]
        return arr

    def step(self, action=None, memory=None, value=None):
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly, and returns
        observation, state, and reward information to the agent.
        :param action: Agent's action to send to environment. Can be a scalar or vector of int/floats.
        :param memory: Vector corresponding to memory used for RNNs, frame-stacking, or other auto-regressive process.
        :param value: Value estimate to send to environment for visualization. Can be a scalar or vector of float(s).
        :return: A Data structure corresponding to the new state of the environment.
        """
        action = {} if action is None else action
        memory = {} if memory is None else memory
        value = {} if value is None else value
        if self._loaded and not self._global_done and self._global_done is not None:
            if isinstance(action, (int, np.int_, float, np.float_, list, np.ndarray)):
                if self._num_external_brains == 1:
                    action = {self._external_brain_names[0]: action}
                elif self._num_external_brains > 1:
                    raise UnityActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names a keys, "
                        "and actions as values".format(self._num_brains))
                else:
                    raise UnityActionException(
                        "There are no external brains in the environment, "
                        "step cannot take an action input")

            if isinstance(memory, (int, np.int_, float, np.float_, list, np.ndarray)):
                if self._num_external_brains == 1:
                    memory = {self._external_brain_names[0]: memory}
                elif self._num_external_brains > 1:
                    raise UnityActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names as keys "
                        "and memories as values".format(self._num_brains))
                else:
                    raise UnityActionException(
                        "There are no external brains in the environment, "
                        "step cannot take a memory input")
            if isinstance(value, (int, np.int_, float, np.float_, list, np.ndarray)):
                if self._num_external_brains == 1:
                    value = {self._external_brain_names[0]: value}
                elif self._num_external_brains > 1:
                    raise UnityActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names as keys "
                        "and state/action value estimates as values".format(self._num_brains))
                else:
                    raise UnityActionException(
                        "There are no external brains in the environment, "
                        "step cannot take a value input")

            for brain_name in list(action.keys()) + list(memory.keys()) + list(value.keys()):
                if brain_name not in self._external_brain_names:
                    raise UnityActionException(
                        "The name {0} does not correspond to an external brain "
                        "in the environment".format(brain_name))

            for b in self._external_brain_names:
                n_agent = len(self._data[b].agents)
                if b not in action:
                    raise UnityActionException("You need to input an action for the brain {0}".format(b))
                action[b] = self._flatten(action[b])
                if b not in memory:
                    memory[b] = [0.0] * self._brains[b].memory_space_size * n_agent
                else:
                    if memory[b] is None:
                        memory[b] = [0.0] * self._brains[b].memory_space_size * n_agent
                    else:
                        memory[b] = self._flatten(memory[b])
                if b not in value:
                    value[b] = [0.0] * n_agent
                else:
                    if value[b] is None:
                        value[b] = [0.0] * n_agent
                    else:
                        value[b] = self._flatten(value[b])
                if not (len(value[b]) == n_agent):
                    raise UnityActionException(
                        "There was a mismatch between the provided value and environment's expectation: "
                        "The brain {0} expected {1} value but was given {2}".format(b, n_agent, len(value[b])))
                if not (len(memory[b]) == self._brains[b].memory_space_size * n_agent):
                    raise UnityActionException(
                        "There was a mismatch between the provided memory and environment's expectation: "
                        "The brain {0} expected {1} memories but was given {2}"
                            .format(b, self._brains[b].memory_space_size * n_agent, len(memory[b])))
                if not ((self._brains[b].action_space_type == "discrete" and len(action[b]) == n_agent) or
                            (self._brains[b].action_space_type == "continuous" and len(
                                action[b]) == self._brains[b].action_space_size * n_agent)):
                    raise UnityActionException(
                        "There was a mismatch between the provided action and environment's expectation: "
                        "The brain {0} expected {1} {2} action(s), but was provided: {3}"
                            .format(b, n_agent if self._brains[b].action_space_type == "discrete" else
                        str(self._brains[b].action_space_size * n_agent), self._brains[b].action_space_type,
                                    str(action[b])))
            self._conn.send(b"STEP")
            self._send_action(action, memory, value)
            return self._get_state()
        elif not self._loaded:
            raise UnityEnvironmentException("No Unity environment is loaded.")
        elif self._global_done:
            raise UnityActionException("The episode is completed. Reset the environment with 'reset()'")
        elif self.global_done is None:
            raise UnityActionException(
                "You cannot conduct step without first calling reset. Reset the environment with 'reset()'")

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._loaded & self._open_socket:
            self._conn.send(b"EXIT")
            self._conn.close()
        if self._open_socket:
            self._socket.close()
            self._loaded = False
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")
