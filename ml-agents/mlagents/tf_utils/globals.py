from typing import Optional

_rank: Optional[int] = None


def get_rank() -> Optional[int]:
    """
    Returns the rank (in the MPI sense) of the current node.
    For local training, this will always be None.
    If this needs to be used, it should be done from outside ml-agents.
    :return:
    """
    return _rank
