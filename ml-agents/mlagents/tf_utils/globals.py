from typing import Optional

_rank: Optional[int] = None
_local_rank: Optional[int] = None


def get_rank() -> Optional[int]:
    return _rank


def broadcast_variables() -> bool:
    return True if _rank is not None else False
