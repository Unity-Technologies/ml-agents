from typing import Dict, Iterator, Any
from collections.abc import Mapping
from mlagents_envs.registry.base_registry_entry import BaseRegistryEntry


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

    _REGISTERED_ENVS: Dict[str, BaseRegistryEntry] = {}

    def register(self, new_entry: BaseRegistryEntry) -> None:
        """
        Registers a new BaseRegistryEntry to the registry. The
        BaseRegistryEntry.identifier value will be used as indexing key.
        If two are more environments are registered under the same key, the most
        recentry added will replace the others.
        """
        self._REGISTERED_ENVS[new_entry.identifier] = new_entry

    def clear(self) -> None:
        """
        Deletes all entries in the registry.
        """
        self._REGISTERED_ENVS.clear()

    def __getitem__(self, identifier: str) -> BaseRegistryEntry:
        """
        Returns the BaseRegistryEntry with the provided identifier. BaseRegistryEntry
        can then be used to make a Unity Environment.
        :param identifier: The identifier of the BaseRegistryEntry
        :returns: The associated BaseRegistryEntry
        """
        if identifier not in self._REGISTERED_ENVS:
            raise KeyError(f"The entry {identifier} is not present in the registry.")
        return self._REGISTERED_ENVS[identifier]

    def __len__(self) -> int:
        return len(self._REGISTERED_ENVS)

    def __iter__(self) -> Iterator[Any]:
        yield from self._REGISTERED_ENVS
