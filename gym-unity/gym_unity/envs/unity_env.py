import logging
import gym
import numpy as np
from mlagents.envs import UnityEnvironment
from gym import error, spaces


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")


class UnityEnv(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    Multi-agent environments use lists for object types, as done here:
    https://github.com/openai/multiagent-particle-envs
    """

    def __init__(self, environment_filename: str, worker_id=0, use_visual=False, multiagent=False):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        :param worker_id: Worker number for environment.
        :param use_visual: Whether to use visual observation or vector observation.
        :param multiagent: Whether to run in multi-agent mode (lists of obs, reward, done).
        """
        self._env = UnityEnvironment(environment_filename, worker_id)
        self.name = self._env.academy_name
        self.visual_obs = None
        self._current_state = None
        self._n_agents = None
        self._multiagent = multiagent

        # Check brain configuration
        if len(self._env.brains) != 1:
            raise UnityGymException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym.")
        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]

        if use_visual and brain.number_visual_observations == 0:
            raise UnityGymException("`use_visual` was set to True, however there are no"
                                    " visual observations as part of this environment.")
        self.use_visual = brain.number_visual_observations >= 1 and use_visual

        if brain.number_visual_observations > 1:
            logger.warning("The environment contains more than one visual observation. "
                           "Please note that only the first will be provided in the observation.")

        if brain.num_stacked_vector_observations != 1:
            raise UnityGymException(
                "There can only be one stacked vector observation in a UnityEnvironment "
                "if it is wrapped in a gym.")

        # Check for number of agents in scene.
        initial_info = self._env.reset()[self.brain_name]
        self._check_agents(len(initial_info.agents))

        # Set observation and action spaces
        if brain.vector_action_space_type == "discrete":
            if len(brain.vector_action_space_size) == 1:
                self._action_space = spaces.Discrete(brain.vector_action_space_size[0])
            else:
                self._action_space = spaces.MultiDiscrete(brain.vector_action_space_size)
        else:
            high = np.array([1] * brain.vector_action_space_size[0])
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([np.inf] * brain.vector_observation_space_size)
        self.action_meanings = brain.vector_action_descriptions
        if self.use_visual:
            if brain.camera_resolutions[0]["blackAndWhite"]:
                depth = 1
            else:
                depth = 3
            self._observation_space = spaces.Box(0, 1, dtype=np.float32,
                                                 shape=(brain.camera_resolutions[0]["height"],
                                                        brain.camera_resolutions[0]["width"],
                                                        depth))
        else:
            self._observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
        info = self._env.reset()[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)

        if not self._multiagent:
            obs, reward, done, info = self._single_step(info)
        else:
            obs, reward, done, info = self._multi_step(info)
        return obs

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        In the case of multi-agent environments, these are lists.
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """

        # Use random actions for all other agents in environment.
        if self._multiagent:
            if not isinstance(action, list):
                raise UnityGymException("The environment was expecting `action` to be a list.")
            if len(action) != self._n_agents:
                raise UnityGymException(
                    "The environment was expecting a list of {} actions.".format(self._n_agents))
            else:
                action = np.array(action)

        info = self._env.step(action)[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self._current_state = info

        if not self._multiagent:
            obs, reward, done, info = self._single_step(info)
        else:
            obs, reward, done, info = self._multi_step(info)
        return obs, reward, done, info

    def _single_step(self, info):
        if self.use_visual:
            self.visual_obs = info.visual_observations[0][0, :, :, :]
            default_observation = self.visual_obs
        else:
            default_observation = info.vector_observations[0, :]

        return default_observation, info.rewards[0], info.local_done[0], {
            "text_observation": info.text_observations[0],
            "brain_info": info}

    def _multi_step(self, info):
        if self.use_visual:
            self.visual_obs = info.visual_observations
            default_observation = self.visual_obs
        else:
            default_observation = info.vector_observations
        return list(default_observation), info.rewards, info.local_done, {
            "text_observation": info.text_observations,
            "brain_info": info}

    def render(self, mode='rgb_array'):
        return self.visual_obs

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def get_action_meanings(self):
        return self.action_meanings

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warn("Could not seed environment %s", self.name)
        return

    def _check_agents(self, n_agents):
        if not self._multiagent and n_agents > 1:
            raise UnityGymException(
                "The environment was launched as a single-agent environment, however"
                "there is more than one agent in the scene.")
        elif self._multiagent and n_agents <= 1:
            raise UnityGymException(
                "The environment was launched as a mutli-agent environment, however"
                "there is only one agent in the scene.")
        if self._n_agents is None:
            self._n_agents = n_agents
            logger.info("{} agents within environment.".format(n_agents))
        elif self._n_agents != n_agents:
            raise UnityGymException("The number of agents in the environment has changed since "
                                    "initialization. This is not supported.")

    @property
    def metadata(self):
        return {'render.modes': ['rgb_array']}

    @property
    def reward_range(self):
        return -float('inf'), float('inf')

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._n_agents
