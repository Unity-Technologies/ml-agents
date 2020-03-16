from typing import Any, Dict, Type
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.components.reward_signals import RewardSignal
from mlagents.trainers.components.reward_signals.extrinsic.signal import (
    ExtrinsicRewardSignal,
)
from mlagents.trainers.components.reward_signals.gail.signal import GAILRewardSignal
from mlagents.trainers.components.reward_signals.curiosity.signal import (
    CuriosityRewardSignal,
)
from mlagents.trainers.policy.tf_policy import TFPolicy


NAME_TO_CLASS: Dict[str, Type[RewardSignal]] = {
    "extrinsic": ExtrinsicRewardSignal,
    "curiosity": CuriosityRewardSignal,
    "gail": GAILRewardSignal,
}


def create_reward_signal(
    policy: TFPolicy, name: str, config_entry: Dict[str, Any]
) -> RewardSignal:
    """
    Creates a reward signal class based on the name and config entry provided as a dict.
    :param policy: The policy class which the reward will be applied to.
    :param name: The name of the reward signal
    :param config_entry: The config entries for that reward signal
    :return: The reward signal class instantiated
    """
    rcls = NAME_TO_CLASS.get(name)
    if not rcls:
        raise UnityTrainerException("Unknown reward signal type {0}".format(name))
    rcls.check_config(config_entry)
    try:
        class_inst = rcls(policy, **config_entry)
    except TypeError:
        raise UnityTrainerException(
            "Unknown parameters given for reward signal {0}".format(name)
        )
    return class_inst
