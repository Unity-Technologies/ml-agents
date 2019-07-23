import attr
from typing import Dict


@attr.s(auto_attribs=True)
class RewardSignalParameters:
    strength: float = 1.0
    gamma: float = .99


@attr.s(auto_attribs=True)
class ExtrinsicRewardSignalParameters(RewardSignalParameters):
    pass

@attr.s(auto_attribs=True)
class CuriosityRewardSignalParameters(RewardSignalParameters):
    encoding_size: int = 256

@attr.s(auto_attribs=True)
class GailRewardSignalParameters(RewardSignalParameters):
    pass


reward_name_to_parameter_class = {
    "extrinsic": ExtrinsicRewardSignalParameters,
    "gail": GailRewardSignalParameters,
    "curiosity": CuriosityRewardSignalParameters
}

@attr.s(auto_attribs=True)
class TrainerParameters:
    trainer: str = "ppo"  # TODO enum
    batch_size: int = 1024
    beta: float = 5.0e-3
    buffer_size: int = 10240
    epsilon: float = 0.2
    hidden_units: int = 128
    lambd: float = 0.95
    learning_rate: float = 3.0e-4
    max_steps: int = 5.0e4
    memory_size: int = 256
    normalize: bool = False
    num_epoch: int = 3
    num_layers: int = 2
    # TODO pretraining?
    time_horizon: int = 64
    sequence_length: int = 64
    summary_freq: int = 1000
    use_recurrent: bool = False
    vis_encode_type: str = "default"  # TODO enum
    reward_signals: Dict[str, RewardSignalParameters] = attr.Factory(dict)

    def __attrs_post_init__(self):
        # Convert the reward_signals dict values
        for reward_name, reward_config in self.reward_signals.items():
            reward_class = reward_name_to_parameter_class[reward_name]
            self.reward_signals[reward_name] = reward_class(**reward_config)
