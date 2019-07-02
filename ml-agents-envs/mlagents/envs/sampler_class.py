import numpy as np 
from abc import ABC, abstractmethod

from .exception import SamplerException

class Sampler(ABC): 

    @abstractmethod
    def sample_parameter(self, *args, **kwargs):
        pass


class UniformSampler(Sampler):
    def __init__(self, param_dict):
        self.check_keys(param_dict)
        self.min_value = param_dict["min"]
        self.max_value = param_dict["max"]


    def check_keys(self, param_dict):
        keys = ["min", "max"]
        for key in keys:
            if key not in param_dict:
                raise SamplerException(
                    key + " is required to use Uniform Sampling but" 
                    " this value isn't provided. Please recheck"
                    " the documentation to ensure that the sampler file"
                    " matches the required format."
                )

    def sample_parameter(self):
        return np.random.uniform(self.min_value, self.max_value)
    

class MultiRangeUniformSampler(Sampler):
    def __init__(self, param_dict):
        self.check_keys(param_dict)
        self.intervals = param_dict["intervals"]
    
    def check_keys(self, param_dict):
        keys = ["intervals"]
        for key in keys:
            if key not in param_dict:
                raise SamplerException(
                    key + " is required to use MultiRange Uniform sampler but" 
                    " this value isn't provided. Please recheck"
                    " the documentation to ensure that the sampler file"
                    " matches the required format."
                )

    
    def sample_parameter(self):
        cur_min, cur_max = self.intervals[np.random.randint(len(self.intervals))]
        return np.random.uniform(cur_min, cur_max)


class GaussianSampler(Sampler):
    def __init__(self, param_dict):
        self.check_keys(param_dict)
        self.mean = param_dict["mean"]
        self.var = param_dict["var"]
    
    def sample_parameter(self):
        return np.random.normal(self.mean, self.var)

    def check_keys(self, param_dict):
        keys = ["mean", "var"]
        for key in keys:
            if key not in param_dict:
                raise SamplerException(
                    key + " is required to use MultiRange Uniform sampler but" 
                    " this value isn't provided. Please recheck"
                    " the documentation to ensure that the sampler file"
                    " matches the required format."
                )



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
    
    def init_sampler_class(name, param_dict):
        sampler_cls = SamplerFactory.NAME_TO_CLASS[name]
        return sampler_cls(param_dict)


class SamplerManager:
    def __init__(self, reset_param_dict):
        self.reset_param_dict = reset_param_dict
        self.sampler_factory = SamplerFactory()
        self.samplers = {}
        for param_name, cur_param_dict in self.reset_param_dict.items():
            # cur_param_dict = reset_param_dict[param_name]
            sampler_name = cur_param_dict["sampler-type"]
            param_sampler = SamplerFactory.init_sampler_class(sampler_name, cur_param_dict)

            self.samplers[param_name] = param_sampler

    def sample_all(self, train_mode = True):
        res = {}
        for param_name, param_sampler in self.samplers.items():
            res[param_name] = param_sampler.sample_parameter()
        return res