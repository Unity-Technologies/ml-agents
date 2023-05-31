from sys import platform
from typing import Optional, Any, List
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import BaseEnv
from mlagents_envs.registry.binary_utils import get_local_binary_path
from mlagents_envs.registry.base_registry_entry import BaseRegistryEntry


class RemoteRegistryEntry(BaseRegistryEntry):
    def __init__(
        self,
        identifier: str,
        expected_reward: Optional[float],
        description: Optional[str],
        linux_url: Optional[str],
        darwin_url: Optional[str],
        win_url: Optional[str],
        additional_args: Optional[List[str]] = None,
        tmp_dir: Optional[str] = None,
    ):
        """
        A RemoteRegistryEntry is an implementation of BaseRegistryEntry that uses a
        Unity executable downloaded from the internet to launch a UnityEnvironment.
        __Note__: The url provided must be a link to a `.zip` file containing a single
        compressed folder with the executable inside. There can only be one executable
        in the folder and it must be at the root of the folder.
        :param identifier: The name of the Unity Environment.
        :param expected_reward: The cumulative reward that an Agent must receive
        for the task to be considered solved.
        :param description: A description of the Unity Environment. Contains human
        readable information about potential special arguments that the make method can
        take as well as information regarding the observation, reward, actions,
        behaviors and number of agents in the Environment.
        :param linux_url: The url of the Unity executable for the Linux platform
        :param darwin_url: The url of the Unity executable for the OSX platform
        :param win_url: The url of the Unity executable for the Windows platform
        """
        super().__init__(identifier, expected_reward, description)
        self._linux_url = linux_url
        self._darwin_url = darwin_url
        self._win_url = win_url
        self._add_args = additional_args
        self._tmp_dir_override = tmp_dir

    def make(self, **kwargs: Any) -> BaseEnv:
        """
        Returns the UnityEnvironment that corresponds to the Unity executable found at
        the provided url. The arguments passed to this method will be passed to the
        constructor of the UnityEnvironment (except for the file_name argument)
        """
        url = None
        if platform == "linux" or platform == "linux2":
            url = self._linux_url
        if platform == "darwin":
            url = self._darwin_url
        if platform == "win32":
            url = self._win_url
        if url is None:
            raise FileNotFoundError(
                f"The entry {self.identifier} does not contain a valid url for this "
                "platform"
            )
        path = get_local_binary_path(
            self.identifier, url, tmp_dir=self._tmp_dir_override
        )
        if "file_name" in kwargs:
            kwargs.pop("file_name")
        args: List[str] = []
        if "additional_args" in kwargs:
            if kwargs["additional_args"] is not None:
                args += kwargs["additional_args"]
        if self._add_args is not None:
            args += self._add_args
        kwargs["additional_args"] = args
        return UnityEnvironment(file_name=path, **kwargs)
