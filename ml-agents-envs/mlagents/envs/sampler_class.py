import numpy as np 
from abc import ABC, abstractmethod

class Sampler(ABC): 

    @abstractmethod
    def sample_parameter(self, *args, **kwargs):
        pass


class UniformSampler(Sampler):
    def __init__(self, param_dict):
        self.min_value = param_dict["min"]
        self.max_value = param_dict["max"]

    def sample_parameter(self):
        return np.random.uniform(self.min_value, self.max_value)
    

class MultiRangeUniformSampler(Sampler):
    def __init__(self, param_dict):
        self.intervals = param_dict["intervals"]
    
    def sample_parameter(self):
        cur_min, cur_max = self.intervals[np.random.randint(len(self.intervals))]
        return np.random.uniform(cur_min, cur_max)


class GaussianSampler(Sampler):
    def __init__(self, param_dict):
        self.mean = param_dict["mean"]
        self.var = param_dict["var"]
    
    def sample_parameter(self):
        return np.random.normal(self.mean, self.var)

class MultiArmBanditSampler(Sampler):
    def sample_parameter(self):
        pass



# To introduce new sampling methods, just need to 'register' them to this sampler factory
class SamplerFactory:
    NAME_TO_CLASS = {
    "uniform": UniformSampler,
    "gaussian": GaussianSampler,
    # "multiarm_bandit": MultiarmBanditSampler,
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

"""
class UniformSampler(Sampler):

    def __init__(self, param_dict, train_mode = True):
        self.param_dict = param_dict
        self.train_mode = train_mode
    
    @staticmethod
    def uniform_sampler(min_val, max_val):
        if (min_val < max_val):
            return np.random.uniform(min_val, max_val)
        else:
            return np.random.uniform(max_val, min_val)


    def sample_parameter(self):
        res = {}
        for key in self.param_dict.keys():
            cur_param_dict = self.param_dict[key]
            rel_min = cur_param_dict["rel_min"]
            rel_max = cur_param_dict["rel_max"]
            if self.train_mode:
                res[key] = self.uniform_sampler(rel_min, rel_max)
            else:
                abs_min = cur_param_dict["abs_min"]
                abs_max = cur_param_dict["abs_max"]
                flip = np.random.random()
                if flip > 0.5:
                    res[key] = self.uniform_sampler(abs_min, rel_min)
                else:
                    res[key] = self.uniform_sampler(rel_max, abs_max)
        return res

    def update_param_info(self, brain_info = None):
        pass

class ParameterSampler:
    def __init__(self, train_sampler, test_sampler):
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler


class SamplerManager:
    def __init__(self, reset_param_dict):
        self.reset_params = reset_param_dict
        self.sampler_factory = SamplerFactory()
        self.samplers = {}
        for param in reset_param_dict.keys():
            train_sampler_type = reset_param_dict[param]["train"]["sampler_type"]
            train_param_dict = reset_param_dict[param]["train"]
            train_sampler = SamplerFactory.init_sampler_class(train_sampler_type, train_param_dict)
            test_sampler_type = reset_param_dict[param]["test"]["sampler_type"]
            test_param_dict = reset_param_dict[param]["test"]
            test_sampler = SamplerFactory.init_sampler_class(test_sampler_type, test_param_dict)
            
            self.samplers[param] = ParameterSampler(train_sampler, test_sampler)

    def sample_all(self, train_mode = True):
        res = {}
        for param_name, param_sampler in self.samplers.items():
            cur_sampler =  param_sampler.train_samper if train_mode else param_sampler.test_sampler
            res[param_name] = cur_sampler.sample()
"""