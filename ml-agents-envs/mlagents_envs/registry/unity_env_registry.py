from typing import Dict, Iterator, Any, List
from collections.abc import Mapping
from mlagents_envs.registry.base_registry_entry import BaseRegistryEntry
from mlagents_envs.registry.binary_utils import (
    load_local_manifest,
    load_remote_manifest,
)
from mlagents_envs.registry.remote_registry_entry import RemoteRegistryEntry


class UnityEnvRegistry(Mapping):
    """
    ### UnityEnvRegistry
    Provides a library of Unity environments that can be launched without the need
    of downloading the Unity Editor.
    The UnityEnvRegistry implements a Map, to access an entry of the Registry, use:
    ```python
    registry = UnityEnvRegistry()
    entry = registry[<environment_identifyier>]
    ```
    An entry has the following properties :
     * `identifier` : Uniquely identifies this environment
     * `expected_reward` : Corresponds to the reward an agent must obtained for the task
     to be considered completed.
     * `description` : A human readable description of the environment.

    To launch a Unity environment from a registry entry, use the `make` method:
    ```python
    registry = UnityEnvRegistry()
    env = registry[<environment_identifyier>].make()
    ```
    """

    def __init__(self):
        self._REGISTERED_ENVS: Dict[str, BaseRegistryEntry] = {}
        self._manifests: List[str] = []
        self._sync = True

    def register(self, new_entry: BaseRegistryEntry) -> None:
        """
        Registers a new BaseRegistryEntry to the registry. The
        BaseRegistryEntry.identifier value will be used as indexing key.
        If two are more environments are registered under the same key, the most
        recentry added will replace the others.
        """
        self._REGISTERED_ENVS[new_entry.identifier] = new_entry

    def register_from_yaml(self, path_to_yaml: str) -> None:
        """
        Registers the environments listed in a yaml file (either local or remote). Note
        that the entries are registered lazily: the registration will only happen when
        an environment is accessed.
        The yaml file must have the following format :
        ```yaml
        environments:
        - <identifier of the first environment>:
            expected_reward: <expected reward of the environment>
            description: | <a multi line description of the environment>
              <continued multi line description>
            linux_url: <The url for the Linux executable zip file>
            darwin_url: <The url for the OSX executable zip file>
            win_url: <The url for the Windows executable zip file>

        - <identifier of the second environment>:
            expected_reward: <expected reward of the environment>
            description: | <a multi line description of the environment>
              <continued multi line description>
            linux_url: <The url for the Linux executable zip file>
            darwin_url: <The url for the OSX executable zip file>
            win_url: <The url for the Windows executable zip file>

        - ...
        ```
        :param path_to_yaml: A local path or url to the yaml file
        """
        self._manifests.append(path_to_yaml)
        self._sync = False

    def _load_all_manifests(self) -> None:
        if not self._sync:
            for path_to_yaml in self._manifests:
                if path_to_yaml[:4] == "http":
                    manifest = load_remote_manifest(path_to_yaml)
                else:
                    manifest = load_local_manifest(path_to_yaml)
                for env in manifest["environments"]:
                    remote_entry_args = list(env.values())[0]
                    remote_entry_args["identifier"] = list(env.keys())[0]
                    self.register(RemoteRegistryEntry(**remote_entry_args))
            self._manifests = []
            self._sync = True

    def clear(self) -> None:
        """
        Deletes all entries in the registry.
        """
        self._REGISTERED_ENVS.clear()
        self._manifests = []
        self._sync = True

    def __getitem__(self, identifier: str) -> BaseRegistryEntry:
        """
        Returns the BaseRegistryEntry with the provided identifier. BaseRegistryEntry
        can then be used to make a Unity Environment.
        :param identifier: The identifier of the BaseRegistryEntry
        :returns: The associated BaseRegistryEntry
        """
        self._load_all_manifests()
        if identifier not in self._REGISTERED_ENVS:
            raise KeyError(f"The entry {identifier} is not present in the registry.")
        return self._REGISTERED_ENVS[identifier]

    def __len__(self) -> int:
        self._load_all_manifests()
        return len(self._REGISTERED_ENVS)

    def __iter__(self) -> Iterator[Any]:
        self._load_all_manifests()
        yield from self._REGISTERED_ENVS


default_registry = UnityEnvRegistry()
default_registry.register_from_yaml(
    "https://storage.googleapis.com/mlagents-test-environments/1.1.0/manifest.yaml"
)  # noqa E501
