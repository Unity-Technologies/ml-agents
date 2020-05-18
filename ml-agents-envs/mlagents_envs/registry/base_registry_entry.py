from abc import abstractmethod
from typing import Any, Optional
from mlagents_envs.base_env import BaseEnv


class BaseRegistryEntry:
    def __init__(
        self,
        identifier: str,
        expected_reward: Optional[float],
        description: Optional[str],
    ):
        """
        BaseRegistryEntry allows launching a Unity Environment with its make method.
        :param identifier: The name of the Unity Environment.
        :param expected_reward: The cumulative reward that an Agent must receive
        for the task to be considered solved.
        :param description: A description of the Unity Environment. Contains human
        readable information about potential special arguments that the make method can
        take as well as information regarding the observation, reward, actions,
        behaviors and number of agents in the Environment.
        """
        self._identifier = identifier
        self._expected_reward = expected_reward
        self._description = description

    @property
    def identifier(self) -> str:
        """
        The unique identifier of the entry
        """
        return self._identifier

    @property
    def expected_reward(self) -> Optional[float]:
        """
        The cumulative reward that an Agent must receive for the task to be considered
        solved.
        """
        return self._expected_reward

    @property
    def description(self) -> Optional[str]:
        """
        A description of the Unity Environment the entry can make.
        """
        return self._description

    @abstractmethod
    def make(self, **kwargs: Any) -> BaseEnv:
        """
        This method creates a Unity BaseEnv (usually a UnityEnvironment).
        """
        raise NotImplementedError(
            f"The make() method not implemented for entry {self.identifier}"
        )
