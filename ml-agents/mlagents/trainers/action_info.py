import numpy as np
from typing import NamedTuple, Any, Dict, Optional, List


class ActionInfo(NamedTuple):
    action: Any
    memory: Any
    text: Any
    value: Any
    outputs: Optional[Dict[str, Any]]


def combine(action_infos: List[ActionInfo]):
    run_outs = [ai.outputs for ai in action_infos]

    run_out = {key: np.array([r[key][0] for r in run_outs]) for key in run_outs[0]
               if type(run_outs[0][key]) is np.ndarray}

    other_keys = [key for key in run_outs[0] if type(run_outs[0][key]) is not np.ndarray]

    for key in other_keys:
        run_out[key] = run_outs[0][key]

    return ActionInfo(
        action=run_out.get('action'),
        memory=run_out.get('memory_out'),
        text=None,
        value=run_out.get('value'),
        outputs=run_out
    )