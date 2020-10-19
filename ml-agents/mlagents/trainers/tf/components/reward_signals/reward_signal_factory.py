from typing import Dict, Type
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.tf.components.reward_signals import RewardSignal
from mlagents.trainers.tf.components.reward_signals.extrinsic.signal import (
    ExtrinsicRewardSignal,
)
from mlagents.trainers.tf.components.reward_signals.gail.signal import GAILRewardSignal
from mlagents.trainers.tf.components.reward_signals.curiosity.signal import (
    CuriosityRewardSignal,
)
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType


NAME_TO_CLASS: Dict[RewardSignalType, Type[RewardSignal]] = {
    RewardSignalType.EXTRINSIC: ExtrinsicRewardSignal,
    RewardSignalType.CURIOSITY: CuriosityRewardSignal,
    RewardSignalType.GAIL: GAILRewardSignal,
}


def create_reward_signal(
    policy: TFPolicy, name: RewardSignalType, settings: RewardSignalSettings
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
        raise UnityTrainerException(f"Unknown reward signal type {name}")

    class_inst = rcls(policy, settings)
    return class_inst
