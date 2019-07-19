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
    # kwargs acts as a sink for extra unneeded args
    def __init__(self, min_value, max_value, **kwargs):
        self.min_value = min_value
        self.max_value = max_value

    def sample_parameter(self) -> float:
        return np.random.uniform(self.min_value, self.max_value)


class MultiRangeUniformSampler(Sampler):
    def __init__(self, intervals, **kwargs):
        self.intervals = intervals
        # Measure the length of the intervals
        interval_lengths = [abs(x[1] - x[0]) for x in self.intervals]
        # Cumulative size of the intervals
        cum_interval_length = sum(interval_lengths)
        # Assign weights to an interval proportionate to the interval size
        self.interval_weights = [x / cum_interval_length for x in interval_lengths]

    def sample_parameter(self) -> float:
        cur_min, cur_max = self.intervals[
            np.random.choice(len(self.intervals), p=self.interval_weights)
        ]
        return np.random.uniform(cur_min, cur_max)


class GaussianSampler(Sampler):
    def __init__(self, mean, var, **kwargs):
        self.mean = mean
        self.var = var

    def sample_parameter(self) -> float:
        return np.random.normal(self.mean, self.var)


# To introduce new sampling methods, just need to 'register' them to this sampler factory
class SamplerFactory:
    NAME_TO_CLASS = {
        "uniform": UniformSampler,
        "gaussian": GaussianSampler,
        "multirange_uniform": MultiRangeUniformSampler,
    }

    @staticmethod
    def register_sampler(name, sampler_cls):
        SamplerFactory.NAME_TO_CLASS[name] = sampler_cls

    @staticmethod
    def init_sampler_class(name, param_dict):
        if name not in SamplerFactory.NAME_TO_CLASS:
            raise SamplerException(
                name + " sampler is not registered in the SamplerFactory."
                " Use the register_sample method to register the string"
                " associated to your sampler in the SamplerFactory."
            )
        sampler_cls = SamplerFactory.NAME_TO_CLASS[name]
        try:
            return sampler_cls(**param_dict)
        except:
            raise SamplerException(
                "The sampler class associated to the " + name + " key in the factory "
                "was not provided the required arguments. Please ensure that the sampler "
                "config file consists of the appropriate keys for this sampler class."
            )


class SamplerManager:
    def __init__(self, reset_param_dict):
        self.reset_param_dict = reset_param_dict if reset_param_dict else {}
        assert isinstance(self.reset_param_dict, dict)
        self.samplers = OrderedDict()
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
        If self.samplers is empty, then bool of it returns false, indicating that the
        sampler manager isn't managing any samplers.
        """
        return not bool(self.samplers)

    def sample_all(self) -> Dict[str, float]:
        res = {}
        for param_name, param_sampler in list(self.samplers.items()):
            res[param_name] = param_sampler.sample_parameter()
        return res
