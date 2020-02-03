from typing import NamedTuple, Any, Dict, List
import numpy as np
from mlagents_envs.base_env import AgentId

ActionInfoOutputs = Dict[str, np.ndarray]


class ActionInfo(NamedTuple):
    action: Any
    value: Any
    outputs: ActionInfoOutputs
    agent_ids: List[AgentId]

    @staticmethod
    def empty() -> "ActionInfo":
        return ActionInfo([], [], {}, [])
