from typing import NamedTuple, Any, Dict, Optional

ActionInfoOutputs = Optional[Dict[str, Any]]


class ActionInfo(NamedTuple):
    action: Any
    text: Any
    value: Any
    outputs: ActionInfoOutputs
