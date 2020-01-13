from typing import NamedTuple, Any, Dict, List
import numpy as np

ActionInfoOutputs = Dict[str, np.ndarray]


class ActionInfo(NamedTuple):
    action: Any
    value: Any
    outputs: ActionInfoOutputs
    agents: List[str]
