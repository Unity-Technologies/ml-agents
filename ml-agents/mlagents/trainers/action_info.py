from typing import NamedTuple, Any, Dict, Optional


class ActionInfo(NamedTuple):
    action: Any
    memory: Any
    text: Any
    value: Any
    outputs: Optional[Dict[str, Any]]
