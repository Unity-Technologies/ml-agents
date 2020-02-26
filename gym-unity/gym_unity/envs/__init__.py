import logging
import itertools
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
from gym import error, spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import BatchedStepResult


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")


GymSingleStepResult = Tuple[np.ndarray, float, bool, Dict]
GymMultiStepResult = Tuple[List[np.ndarray], List[float], List[bool], Dict]
GymStepResult = Union[GymSingleStepResult, GymMultiStepResult]


class UnityEnv(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    Multi-agent environments use lists for object types, as done here:
    https://github.com/openai/multiagent-particle-envs
    """

    def __init__(
        self,
        environment_filename: str,
        worker_id: int = 0,
        use_visual: bool = False,
        uint8_visual: bool = False,
        multiagent: bool = False,
        flatten_branched: bool = False,
        no_graphics: bool = False,
        allow_multiple_visual_obs: bool = False,
    ):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        :param worker_id: Worker number for environment.
        :param use_visual: Whether to use visual observation or vector observation.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param multiagent: Whether to run in multi-agent mode (lists of obs, reward, done).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param no_graphics: Whether to run the Unity simulator in no-graphics mode
        :param allow_multiple_visual_obs: If True, return a list of visual observations instead of only one.
        """
        base_port = 5005
        if environment_filename is None:
            base_port = UnityEnvironment.DEFAULT_EDITOR_PORT

        self._env = UnityEnvironment(
            environment_filename,
            worker_id,
            base_port=base_port,
            no_graphics=no_graphics,
        )

        # Take a single step so that the brain information will be sent over
        if not self._env.get_agent_groups():
            self._env.step()

        self.visual_obs = None
        self._n_agents = -1

        self.agent_mapper = AgentIdIndexMapper()

        # Save the step result from the last time all Agents requested decisions.
        self._previous_step_result: BatchedStepResult = None
        self._multiagent = multiagent
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_visual_obs = allow_multiple_visual_obs

        # Check brain configuration
        if len(self._env.get_agent_groups()) != 1:
            raise UnityGymException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym."
            )

        self.brain_name = self._env.get_agent_groups()[0]
        self.name = self.brain_name
        self.group_spec = self._env.get_agent_group_spec(self.brain_name)

        if use_visual and self._get_n_vis_obs() == 0:
            raise UnityGymException(
                "`use_visual` was set to True, however there are no"
                " visual observations as part of this environment."
            )
        self.use_visual = self._get_n_vis_obs() >= 1 and use_visual

        if not use_visual and uint8_visual:
            logger.warning(
                "`uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual

        if self._get_n_vis_obs() > 1 and not self._allow_multiple_visual_obs:
            logger.warning(
                "The environment contains more than one visual observation. "
                "You must define allow_multiple_visual_obs=True to received them all. "
                "Otherwise, please note that only the first will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        step_result = self._env.get_step_result(self.brain_name)
        self._check_agents(step_result.n_agents())
        self._previous_step_result = step_result
        self.agent_mapper.set_initial_agents(list(self._previous_step_result.agent_id))

        # Set observation and action spaces
        if self.group_spec.is_action_discrete():
            branches = self.group_spec.discrete_action_branches
            if self.group_spec.action_shape == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        else:
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            high = np.array([1] * self.group_spec.action_shape)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([np.inf] * self._get_vec_obs_size())
        if self.use_visual:
            shape = self._get_vis_obs_shape()
            if uint8_visual:
                self._observation_space = spaces.Box(
                    0, 255, dtype=np.uint8, shape=shape
                )
            else:
                self._observation_space = spaces.Box(
                    0, 1, dtype=np.float32, shape=shape
                )

        else:
            self._observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        step_result = self._step(True)
        n_agents = step_result.n_agents()
        self._check_agents(n_agents)
        self.game_over = False

        if not self._multiagent:
            res: GymStepResult = self._single_step(step_result)
        else:
            res = self._multi_step(step_result)
        return res[0]

    def step(self, action: List[Any]) -> GymStepResult:
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
            info (dict): contains auxiliary diagnostic information, including BatchedStepResult.
        """

        # Use random actions for all other agents in environment.
        if self._multiagent:
            if not isinstance(action, list):
                raise UnityGymException(
                    "The environment was expecting `action` to be a list."
                )
            if len(action) != self._n_agents:
                raise UnityGymException(
                    "The environment was expecting a list of {} actions.".format(
                        self._n_agents
                    )
                )
            else:
                if self._flattener is not None:
                    # Action space is discrete and flattened - we expect a list of scalars
                    action = [self._flattener.lookup_action(_act) for _act in action]
                action = np.array(action)
        else:
            if self._flattener is not None:
                # Translate action into list
                action = self._flattener.lookup_action(action)

        spec = self.group_spec
        action = np.array(action).reshape((self._n_agents, spec.action_size))
        action = self._sanitize_action(action)
        self._env.set_actions(self.brain_name, action)

        step_result = self._step()

        n_agents = step_result.n_agents()
        self._check_agents(n_agents)

        if not self._multiagent:
            single_res = self._single_step(step_result)
            self.game_over = single_res[2]
            return single_res
        else:
            multi_res = self._multi_step(step_result)
            self.game_over = all(multi_res[2])
            return multi_res

    def _single_step(self, info: BatchedStepResult) -> GymSingleStepResult:
        if self.use_visual:
            visual_obs = self._get_vis_obs_list(info)

            if self._allow_multiple_visual_obs:
                visual_obs_list = []
                for obs in visual_obs:
                    visual_obs_list.append(self._preprocess_single(obs[0]))
                self.visual_obs = visual_obs_list
            else:
                self.visual_obs = self._preprocess_single(visual_obs[0][0])

            default_observation = self.visual_obs
        elif self._get_vec_obs_size() > 0:
            default_observation = self._get_vector_obs(info)[0, :]
        else:
            raise UnityGymException(
                "The Agent does not have vector observations and the environment was not setup "
                + "to use visual observations."
            )

        return (
            default_observation,
            info.reward[0],
            info.done[0],
            {"batched_step_result": info},
        )

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _multi_step(self, info: BatchedStepResult) -> GymMultiStepResult:
        if self.use_visual:
            self.visual_obs = self._preprocess_multi(self._get_vis_obs_list(info))
            default_observation = self.visual_obs
        else:
            default_observation = self._get_vector_obs(info)
        return (
            list(default_observation),
            list(info.reward),
            list(info.done),
            {"batched_step_result": info},
        )

    def _get_n_vis_obs(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> Optional[Tuple]:
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                return shape
        return None

    def _get_vis_obs_list(self, step_result: BatchedStepResult) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(self, step_result: BatchedStepResult) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result

    def _preprocess_multi(
        self, multiple_visual_obs: List[np.ndarray]
    ) -> List[np.ndarray]:
        if self.uint8_visual:
            return [
                (255.0 * _visual_obs).astype(np.uint8)
                for _visual_obs in multiple_visual_obs
            ]
        else:
            return multiple_visual_obs

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment %s", self.name)
        return

    def _check_agents(self, n_agents: int) -> None:
        if not self._multiagent and n_agents > 1:
            raise UnityGymException(
                "The environment was launched as a single-agent environment, however "
                "there is more than one agent in the scene."
            )
        elif self._multiagent and n_agents <= 1:
            raise UnityGymException(
                "The environment was launched as a mutli-agent environment, however "
                "there is only one agent in the scene."
            )
        if self._n_agents == -1:
            self._n_agents = n_agents
            logger.info("{} agents within environment.".format(n_agents))
        elif self._n_agents != n_agents:
            raise UnityGymException(
                "The number of agents in the environment has changed since "
                "initialization. This is not supported."
            )

    def _sanitize_info(self, step_result: BatchedStepResult) -> BatchedStepResult:
        n_extra_agents = step_result.n_agents() - self._n_agents
        if n_extra_agents < 0 or n_extra_agents > self._n_agents:
            # In this case, some Agents did not request a decision when expected
            # or too many requested a decision
            raise UnityGymException(
                "The number of agents in the scene does not match the expected number."
            )

        if step_result.n_agents() - sum(step_result.done) != self._n_agents:
            raise UnityGymException(
                "The number of agents in the scene does not match the expected number."
            )

        for index, agent_id in enumerate(step_result.agent_id):
            if step_result.done[index]:
                self.agent_mapper.mark_agent_done(agent_id, step_result.reward[index])

        # Set the new AgentDone flags to True
        # Note that the corresponding agent_id that gets marked done will be different
        # than the original agent that was done, but this is OK since the gym interface
        # only cares about the ordering.
        for index, agent_id in enumerate(step_result.agent_id):
            if not self._previous_step_result.contains_agent(agent_id):
                # Register this agent, and get the reward of the previous agent that
                # was in its index, so that we can return it to the gym.
                last_reward = self.agent_mapper.register_new_agent_id(agent_id)
                step_result.done[index] = True
                step_result.reward[index] = last_reward

        self._previous_step_result = step_result  # store the new original

        # Get a permutation of the agent IDs so that a given ID stays in the same
        # index as where it was first seen.
        new_id_order = self.agent_mapper.get_id_permutation(list(step_result.agent_id))

        _mask: Optional[List[np.array]] = None
        if step_result.action_mask is not None:
            _mask = []
            for mask_index in range(len(step_result.action_mask)):
                _mask.append(step_result.action_mask[mask_index][new_id_order])
        new_obs: List[np.array] = []
        for obs_index in range(len(step_result.obs)):
            new_obs.append(step_result.obs[obs_index][new_id_order])
        return BatchedStepResult(
            obs=new_obs,
            reward=step_result.reward[new_id_order],
            done=step_result.done[new_id_order],
            max_step=step_result.max_step[new_id_order],
            agent_id=step_result.agent_id[new_id_order],
            action_mask=_mask,
        )

    def _sanitize_action(self, action: np.array) -> np.array:
        sanitized_action = np.zeros(
            (self._previous_step_result.n_agents(), self.group_spec.action_size)
        )
        for index, agent_id in enumerate(self._previous_step_result.agent_id):
            if not self._previous_step_result.done[index]:
                array_index = self.agent_mapper.get_gym_index(agent_id)
                sanitized_action[index, :] = action[array_index, :]
        return sanitized_action

    def _step(self, needs_reset: bool = False) -> BatchedStepResult:
        if needs_reset:
            self._env.reset()
        else:
            self._env.step()
        info = self._env.get_step_result(self.brain_name)
        # Two possible cases here:
        # 1) all agents requested decisions (some of which might be done)
        # 2) some Agents were marked Done in between steps.
        # In case 2,  we re-request decisions until all agents request a real decision.
        while info.n_agents() - sum(info.done) < self._n_agents:
            if not info.done.all():
                raise UnityGymException(
                    "The environment does not have the expected amount of agents. "
                    + "Some agents did not request decisions at the same time."
                )
            for agent_id, reward in zip(info.agent_id, info.reward):
                self.agent_mapper.mark_agent_done(agent_id, reward)

            self._env.step()
            info = self._env.get_step_result(self.brain_name)
        return self._sanitize_info(info)

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

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


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]


class AgentIdIndexMapper:
    def __init__(self) -> None:
        self._agent_id_to_gym_index: Dict[int, int] = {}
        self._done_agents_index_to_last_reward: Dict[int, float] = {}

    def set_initial_agents(self, agent_ids: List[int]) -> None:
        """
        Provide the initial list of agent ids for the mapper
        """
        for idx, agent_id in enumerate(agent_ids):
            self._agent_id_to_gym_index[agent_id] = idx

    def mark_agent_done(self, agent_id: int, reward: float) -> None:
        """
        Declare the agent done with the corresponding final reward.
        """
        gym_index = self._agent_id_to_gym_index.pop(agent_id)
        self._done_agents_index_to_last_reward[gym_index] = reward

    def register_new_agent_id(self, agent_id: int) -> float:
        """
        Adds the new agent ID and returns the reward to use for the previous agent in this index
        """
        # Any free index is OK here.
        free_index, last_reward = self._done_agents_index_to_last_reward.popitem()
        self._agent_id_to_gym_index[agent_id] = free_index
        return last_reward

    def get_id_permutation(self, agent_ids: List[int]) -> List[int]:
        """
        Get the permutation from new agent ids to the order that preserves the positions of previous agents.
        The result is a list with each integer from 0 to len(_agent_id_to_gym_index)-1
        appearing exactly once.
        """
        # Map the new agent ids to the their index
        new_agent_ids_to_index = {
            agent_id: idx for idx, agent_id in enumerate(agent_ids)
        }

        # Make the output list. We don't write to it sequentially, so start with dummy values.
        new_permutation = [-1] * len(self._agent_id_to_gym_index)

        # For each agent ID, find the new index of the agent, and write it in the original index.
        for agent_id, original_index in self._agent_id_to_gym_index.items():
            new_permutation[original_index] = new_agent_ids_to_index[agent_id]
        return new_permutation

    def get_gym_index(self, agent_id: int) -> int:
        """
        Get the gym index for the current agent.
        """
        return self._agent_id_to_gym_index[agent_id]


class AgentIdIndexMapperSlow:
    """
    Reference implementation of AgentIdIndexMapper.
    The operations are O(N^2) so it shouldn't be used for large numbers of agents.
    See AgentIdIndexMapper for method descriptions
    """

    def __init__(self) -> None:
        self._gym_id_order: List[int] = []
        self._done_agents_index_to_last_reward: Dict[int, float] = {}

    def set_initial_agents(self, agent_ids: List[int]) -> None:
        self._gym_id_order = list(agent_ids)

    def mark_agent_done(self, agent_id: int, reward: float) -> None:
        gym_index = self._gym_id_order.index(agent_id)
        self._done_agents_index_to_last_reward[gym_index] = reward
        self._gym_id_order[gym_index] = -1

    def register_new_agent_id(self, agent_id: int) -> float:
        original_index = self._gym_id_order.index(-1)
        self._gym_id_order[original_index] = agent_id
        reward = self._done_agents_index_to_last_reward.pop(original_index)
        return reward

    def get_id_permutation(self, agent_ids):
        new_id_order = []
        for agent_id in self._gym_id_order:
            new_id_order.append(agent_ids.index(agent_id))
        return new_id_order

    def get_gym_index(self, agent_id: int) -> int:
        return self._gym_id_order.index(agent_id)
