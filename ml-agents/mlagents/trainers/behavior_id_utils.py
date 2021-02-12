from typing import NamedTuple, Optional
from urllib.parse import urlparse, parse_qs


class BehaviorIdentifiers(NamedTuple):
    """
    BehaviorIdentifiers is a named tuple of the identifiers that uniquely distinguish
    an agent encountered in the trainer_controller. The named tuple consists of the
    fully qualified behavior name, the name of the brain name (corresponds to trainer
    in the trainer controller) and the team id.  In the future, this can be extended
    to support further identifiers.
    """

    behavior_id: str
    brain_name: str
    team_id: int
    group_id: int

    @staticmethod
    def from_name_behavior_id(name_behavior_id: str) -> "BehaviorIdentifiers":
        """
        Parses a name_behavior_id of the form name?team=0&group=0
        into a BehaviorIdentifiers NamedTuple.
        This allows you to access the brain name and team id of an agent
        :param name_behavior_id: String of behavior params in HTTP format.
        :returns: A BehaviorIdentifiers object.
        """
        parsed = urlparse(name_behavior_id)
        name = parsed.path
        ids = parse_qs(parsed.query)
        team_id: int = 0
        group_id: int = 0
        if "team" in ids:
            team_id = int(ids["team"][0])
        if "group" in ids:
            group_id = int(ids["group"][0])
        return BehaviorIdentifiers(
            behavior_id=name_behavior_id,
            brain_name=name,
            team_id=team_id,
            group_id=group_id,
        )


def create_name_behavior_id(
    name: str, team_id: Optional[int] = None, group_id: Optional[int] = None
) -> str:
    """
    Reconstructs fully qualified behavior name from name and team_id
    :param name: brain name
    :param team_id: team ID
    :return: name_behavior_id
    """
    final_name = name
    if team_id is not None:
        final_name += f"?team={team_id}"
    if group_id is not None:
        final_name += f"&group={group_id}"
    return final_name


def get_global_agent_id(worker_id: int, agent_id: int) -> str:
    """
    Create an agent id that is unique across environment workers using the worker_id.
    """
    return f"${worker_id}-{agent_id}"


def get_global_group_id(worker_id: int, group_id: int) -> str:
    """
    Create an agent id that is unique across environment workers using the worker_id.
    """
    return f"${worker_id}-{group_id}"
