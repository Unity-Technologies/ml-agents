from typing import NamedTuple
from urllib.parse import urlparse, parse_qs


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

        parsed = urlparse(name_behavior_id)
        name = parsed.path
        ids = parse_qs(parsed.query)
        team_id: int = 0
        if "team" in ids:
            team_id = int(ids["team"][0])
        return BehaviorIdentifiers(
            behavior_id=name_behavior_id, brain_name=name, team_id=team_id
        )
