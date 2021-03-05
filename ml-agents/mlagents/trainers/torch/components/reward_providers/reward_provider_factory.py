from typing import Dict, Type
from mlagents.trainers.exception import UnityTrainerException

from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType

from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.torch.components.reward_providers.extrinsic_reward_provider import (
    ExtrinsicRewardProvider,
)
from mlagents.trainers.torch.components.reward_providers.curiosity_reward_provider import (
    CuriosityRewardProvider,
)
from mlagents.trainers.torch.components.reward_providers.gail_reward_provider import (
    GAILRewardProvider,
)
from mlagents.trainers.torch.components.reward_providers.rnd_reward_provider import (
    RNDRewardProvider,
)

from mlagents_envs.base_env import BehaviorSpec

NAME_TO_CLASS: Dict[RewardSignalType, Type[BaseRewardProvider]] = {
    RewardSignalType.EXTRINSIC: ExtrinsicRewardProvider,
    RewardSignalType.CURIOSITY: CuriosityRewardProvider,
    RewardSignalType.GAIL: GAILRewardProvider,
    RewardSignalType.RND: RNDRewardProvider,
}


def create_reward_provider(
    name: RewardSignalType, specs: BehaviorSpec, settings: RewardSignalSettings
) -> BaseRewardProvider:
    """
    Creates a reward provider class based on the name and config entry provided as a dict.
    :param name: The name of the reward signal
    :param specs: The BehaviorSpecs of the policy
    :param settings: The RewardSignalSettings for that reward signal
    :return: The reward signal class instantiated
    """
    rcls = NAME_TO_CLASS.get(name)
    if not rcls:
        raise UnityTrainerException(f"Unknown reward signal type {name}")

    class_inst = rcls(specs, settings)
    return class_inst
