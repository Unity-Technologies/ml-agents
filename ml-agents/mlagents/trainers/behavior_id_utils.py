from typing import NamedTuple


class BehaviorIdentifiers(NamedTuple):
    behavior_id: str
    brain_name: str
    team_id: int

    @staticmethod
    def from_name_behavior_id(name_behavior_id: str) -> "BehaviorIdentifiers":
        """
        Parses a name_behavior_id of the form name?team=0&param1=i&...
        into a BehaviorIdentifiers NamedTuple.
        This allows you to access the brain name and distinguishing identifiers
        without parsing more than once.
        :param name_behavior_id: String of behavior params in HTTP format.
        :returns: A BehaviorIdentifiers object.
        """

        team_id: int = 0
        if "?" in name_behavior_id:
            name, team_and_id = name_behavior_id.rsplit("?", 1)
            _, team_id_str = team_and_id.split("=")
            team_id = int(team_id_str)
        else:
            name = name_behavior_id

        return BehaviorIdentifiers(
            behavior_id=name_behavior_id, brain_name=name, team_id=team_id
        )
