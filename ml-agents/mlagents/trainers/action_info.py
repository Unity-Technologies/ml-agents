from typing import NamedTuple, Any, Dict

ActionInfoOutputs = Dict[str, Any]


class ActionInfo(NamedTuple):
    action: Any
    value: Any
    outputs: ActionInfoOutputs
