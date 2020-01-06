from typing import Dict, List, NamedTuple

class BehaviorIdentifiers(NamedTuple):
    brain_name: str
    behavior_ids: Dict[str, int]

    @staticmethod
    def from_name_behavior_id(name_behavior_id: str) -> "BehaviorIdentifiers":
        """
        Parses a name_behavior_id into a BehaviorIdentifiers NamedTuple.
        This allows you to access the brain name and distinguishing identifiers
        without parsing more than once.
        :param name_behavior_id: String of behavior params in HTTP format.
        :returns: A BehaviorIdentifiers object.
        """

        ids: Dict[str, int] = {}
        if '?' in name_behavior_id:
            name, identifiers = name_behavior_id.split("?")
            if '&' in identifiers:
                identifiers = identifiers.split("&")
            else:
                identifiers = [identifiers]

            for identifier in identifiers:
                key, value = identifier.split("=")
                ids[key] = value
        else:
            name = name_behavior_id   

        return BehaviorIdentifiers(brain_name=name, behavior_ids=ids)

class Cycle():
    def __init__(self):
        self.iterable : List[str] = []
        self.counter : int = 0

    def add(self, name: str) -> None:
        self.iterable.append(name)

    def get(self) -> str:
        name = self.iterable[self.counter]
        self.counter = (self.counter + 1) % len(self.iterable)
        return name
