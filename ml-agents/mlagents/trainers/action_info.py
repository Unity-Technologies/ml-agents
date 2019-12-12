from typing import NamedTuple, Any, Dict
import numpy as np

ActionInfoOutputs = Dict[str, np.ndarray]


class ActionInfo(NamedTuple):
    action: Any
    value: Any
    outputs: ActionInfoOutputs
