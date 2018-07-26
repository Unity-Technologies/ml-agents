import gym
import numpy as np
from unityagents import UnityEnvironment
from gym import error, spaces, logger


class UnityGymWrapperException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass


def multi_agent_check(info):
    if len(info.agents) != 1:
        logger.warn("Environment contains multiple agents. Only first agent is controllable via gym interface.")


class UnityGymEnv(gym.Env):
    def __init__(self, environment_filename: str, worker_id=0, default_visual=True):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym
        """
        self._env = UnityEnvironment(environment_filename, worker_id)
        self.name = self._env.academy_name
        self.visual_obs = None
        self._current_state = None
        if len(self._env.brains) != 1:
            raise UnityGymWrapperException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym.")
        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]
        self.use_visual = brain.number_visual_observations == 1 and default_visual
        if brain.num_stacked_vector_observations != 1:
            raise UnityGymWrapperException(
                "There can only be one stacked vector observation in a UnityEnvironment "
                "if it is wrapped in a gym.")
        if brain.vector_action_space_type == "discrete":
            self._action_space = spaces.Discrete(brain.vector_action_space_size)
        else:
            high = np.array([1] * brain.vector_action_space_size)
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

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """

        # Use random actions for all other agents in environment.
        if self._env.brains[self.brain_name].vector_action_space_type == 'continuous':
            all_actions = np.random.randn(len(self._current_state.agents),
                                          self._env.brains[self.brain_name].vector_action_space_size)
        else:
            all_actions = np.random.randint(0, self._env.brains[self.brain_name].vector_action_space_size,
                                            size=(len(self._current_state.agents)))

        all_actions[0, :] = action
        info = self._env.step(all_actions)[self.brain_name]
        self._current_state = info
        if self.use_visual:
            self.visual_obs = info.visual_observations[0][0, :, :, :]
            default_observation = self.visual_obs
        else:
            default_observation = info.vector_observations[0, :]

        return default_observation, info.rewards[0], \
               info.local_done[0], {"text_observation": info.text_observations[0], "brain_info": info}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        info = self._env.reset()[self.brain_name]
        multi_agent_check(info)
        if self.use_visual:
            self.visual_obs = info.visual_observations[0][0, :, :, :]
            default_observation = self.visual_obs
        else:
            default_observation = info.vector_observations[0, :]
        self._current_state = info
        return default_observation

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
