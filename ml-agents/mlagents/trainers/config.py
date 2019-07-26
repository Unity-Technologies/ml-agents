import attr
from functools import partial
from typing import Any, Callable, Dict, TypeVar, Type

T = TypeVar("T")


def generate_subclass_mapping(base_cls):
    """
    Takes a base class, and constructs a mapping from the names of its sublclasses to the subclass type. For example

        class IceCream:
            pass

        class ChocolateIceCream(IceCream):
            pass

        class VanillaIceCream(IceCream):
            pass

        >>> generate_subclass_mapping(IceCream)
        {"chocolate": ChocolateIceCream, "vanilla": VanillaIceCream}

    The keys are derived by removing the parent class name from the subclass names and converting to lowercase
    TODO - allow class level overrides for the keys, e.g.
        class RockyRoadIceCream(IceCream):
            CONFIG_NAME = "rocky_road"
    """
    subclasses = base_cls.__subclasses__()
    subclass_mapping = {}
    for sub_cls in subclasses:
        short_name = sub_cls.__name__.replace(base_cls.__name__, "").lower()
        # TODO check sub_cls.CONFIG_NAME for overrides
        subclass_mapping[short_name] = sub_cls
    return subclass_mapping


def map_subclasses_dict(base_cls: T, configs: Dict[str, Any]) -> Dict[str, T]:
    """
    Given a base class and a nested dictionary, where the keys follow the naming convention for subclasses in
    generate_subclass_mapping(), convert the values to the corresponding subclass instances
    :param base_cls:
    :param configs:
    :return:
    """
    res = {}
    subclass_mapping = generate_subclass_mapping(base_cls)
    for short_name, config in configs.items():
        cls = subclass_mapping[short_name]
        instance = cls(**config)
        res[short_name] = instance
    return res


@attr.s(auto_attribs=True)
class RewardSignalParameters:
    strength: float = 1.0
    gamma: float = 0.99


@attr.s(auto_attribs=True)
class ExtrinsicRewardSignalParameters(RewardSignalParameters):
    pass


@attr.s(auto_attribs=True)
class CuriosityRewardSignalParameters(RewardSignalParameters):
    encoding_size: int = 256


@attr.s(auto_attribs=True)
class GailRewardSignalParameters(RewardSignalParameters):
    pass


# Ideally we'd use functools.partial to create a single-argument callable for converting the RewardSignal inputs
# However mypy doesn't seem to like this.
# convert_reward_signal_parameters = partial(map_subclasses_dict, RewardSignalParameters)
def convert_reward_signal_parameters(
    configs: Dict[str, Any]
) -> Dict[str, Type[RewardSignalParameters]]:
    return map_subclasses_dict(RewardSignalParameters, configs)


def int_float(x: Any) -> int:
    """
    Converts a string in scientific notation e.g. "5.0e4" to an int
    """
    return int(float(x))


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
    max_steps: int = attr.ib(default=int(5.0e4), converter=int_float)
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
    reward_signals: Dict[str, RewardSignalParameters] = attr.ib(
        converter=convert_reward_signal_parameters, factory=dict
    )
