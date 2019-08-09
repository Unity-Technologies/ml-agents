import numpy as np
from typing import *
from functools import *
from collections import OrderedDict
from abc import ABC, abstractmethod

from .exception import SamplerException


class Sampler(ABC):
    @abstractmethod
    def sample_parameter(self) -> float:
        pass


class UniformSampler(Sampler):
    """
    Uniformly draws a single sample in the range [min_value, max_value).
    """

    def __init__(
        self, min_value: Union[int, float], max_value: Union[int, float], **kwargs
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def sample_parameter(self) -> float:
        return np.random.uniform(self.min_value, self.max_value)


class MultiRangeUniformSampler(Sampler):
    """
    Draws a single sample uniformly from the intervals provided. The sampler
    first picks an interval based on a weighted selection, with the weights
    assigned to an interval based on its range. After picking the range,
    it proceeds to pick a value uniformly in that range.
    """

    def __init__(self, intervals: List[List[Union[int, float]]], **kwargs) -> None:
        self.intervals = intervals
        # Measure the length of the intervals
        interval_lengths = [abs(x[1] - x[0]) for x in self.intervals]
        cum_interval_length = sum(interval_lengths)
        # Assign weights to an interval proportionate to the interval size
        self.interval_weights = [x / cum_interval_length for x in interval_lengths]

    def sample_parameter(self) -> float:
        cur_min, cur_max = self.intervals[
            np.random.choice(len(self.intervals), p=self.interval_weights)
        ]
        return np.random.uniform(cur_min, cur_max)


class GaussianSampler(Sampler):
    """
    Draw a single sample value from a normal (gaussian) distribution.
    This sampler is characterized by the mean and the standard deviation.
    """

    def __init__(
        self, mean: Union[float, int], st_dev: Union[float, int], **kwargs
    ) -> None:
        self.mean = mean
        self.st_dev = st_dev

    def sample_parameter(self) -> float:
        return np.random.normal(self.mean, self.st_dev)


class SamplerFactory:
    """
    Maintain a directory of all samplers available.
    Add new samplers using the register_sampler method.
    """

    NAME_TO_CLASS = {
        "uniform": UniformSampler,
        "gaussian": GaussianSampler,
        "multirange_uniform": MultiRangeUniformSampler,
    }

    @staticmethod
    def register_sampler(name: str, sampler_cls: Type[Sampler]) -> None:
        SamplerFactory.NAME_TO_CLASS[name] = sampler_cls

    @staticmethod
    def init_sampler_class(name: str, params: Dict[str, Any]):
        if name not in SamplerFactory.NAME_TO_CLASS:
            raise SamplerException(
                name + " sampler is not registered in the SamplerFactory."
                " Use the register_sample method to register the string"
                " associated to your sampler in the SamplerFactory."
            )
        sampler_cls = SamplerFactory.NAME_TO_CLASS[name]
        try:
            return sampler_cls(**params)
        except TypeError:
            raise SamplerException(
                "The sampler class associated to the " + name + " key in the factory "
                "was not provided the required arguments. Please ensure that the sampler "
                "config file consists of the appropriate keys for this sampler class."
            )


class SamplerManager:
    def __init__(self, reset_param_dict: Dict[str, Any]) -> None:
        self.reset_param_dict = reset_param_dict if reset_param_dict else {}
        assert isinstance(self.reset_param_dict, dict)
        self.samplers: Dict[str, Sampler] = {}
        for param_name, cur_param_dict in self.reset_param_dict.items():
            if "sampler-type" not in cur_param_dict:
                raise SamplerException(
                    "'sampler_type' argument hasn't been supplied for the {0} parameter".format(
                        param_name
                    )
                )
            sampler_name = cur_param_dict.pop("sampler-type")
            param_sampler = SamplerFactory.init_sampler_class(
                sampler_name, cur_param_dict
            )

            self.samplers[param_name] = param_sampler

    def is_empty(self) -> bool:
        """
        Check for if sampler_manager is empty.
        """
        return not bool(self.samplers)

    def sample_all(self) -> Dict[str, float]:
        res = {}
        for param_name, param_sampler in list(self.samplers.items()):
            res[param_name] = param_sampler.sample_parameter()
        return res
