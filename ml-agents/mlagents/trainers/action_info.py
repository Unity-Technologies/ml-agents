from typing import NamedTuple, Any, Dict
from numpy import np

ActionInfoOutputs = Dict[str, np.ndarray]


class ActionInfo(NamedTuple):
    action: Any
    value: Any
    outputs: ActionInfoOutputs
