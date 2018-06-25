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
        raise UnityGymWrapperException(
            "There can only be one agent in a UnityEnvironment "
            "if it is wrapped in a gym.")


class GymWrapper(gym.Env):
    def __init__(self, unity_environment: UnityEnvironment):
        """
        Environment initialization
        :param unity_environment: The UnityEnvironment to be wrapped for gym
        """
        self.env = unity_environment
        self.name = self.env.academy_name
        self.visual_obs = None
        if len(self.env.brains) != 1:
            raise UnityGymWrapperException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym.")
        self.brain_name = self.env.external_brain_names[0]
        brain = self.env.brains[self.brain_name]
        self.use_visual = brain.number_visual_observations == 1
        if brain.num_stacked_vector_observations != 1:
            raise UnityGymWrapperException(
                "There can only be one stacked vector observation in a UnityEnvironment "
                "if it is wrapped in a gym.")
        if brain.vector_action_space_type == "discrete":
            self._action_space = spaces.Discrete(brain.vector_action_space_size)
        elif brain.vector_action_space_type == "continuous":
            high = np.array([1] * brain.vector_action_space_size)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            raise UnityGymWrapperException("Could not recognize the vector observation space type")
        high = np.array([np.inf] * brain.vector_observation_space_size)
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
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        info = self.env.step(action)[self.brain_name]
        multi_agent_check(info)
        if self.use_visual:
            self.visual_obs = info.visual_observations[0][0, :, :, :]
        return info.vector_observations[0, :], info.rewards[0], info.local_done[0], \
               {"text_observation": info.text_observations[0]}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        info = self.env.reset()[self.brain_name]
        multi_agent_check(info)
        if self.use_visual:
            self.visual_obs = info.visual_observations[0][0, :, :, :]
        return info.vector_observations[0, :]

    def render(self, mode='rgb_array'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
\        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        return self.visual_obs

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.env.close()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
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
