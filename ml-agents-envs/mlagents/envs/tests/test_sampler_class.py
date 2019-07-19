from math import isclose
import pytest

from mlagents.envs.sampler_class import SamplerManager
from mlagents.envs.sampler_class import (
    UniformSampler,
    MultiRangeUniformSampler,
    GaussianSampler,
)
from mlagents.envs.exception import UnityException


def basic_3Dball_sampler():
    return {
        "mass": {"sampler-type": "uniform", "min_value": 5, "max_value": 10},
        "gravity": {
            "sampler-type": "multirange_uniform",
            "intervals": [[8, 11], [15, 20]],
        },
    }


def check_value_in_intervals(val, intervals):
    check_in_bounds = [a <= val <= b for a, b in intervals]
    return any(check_in_bounds)


def test_3Dball_sampler():
    config = basic_3Dball_sampler()
    sampler = SamplerManager(config)

    assert sampler.is_empty() is False
    assert isinstance(sampler.samplers["mass"], UniformSampler)
    assert isinstance(sampler.samplers["gravity"], MultiRangeUniformSampler)

    cur_sample = sampler.sample_all()

    # Check uniform sampler for mass
    assert sampler.samplers["mass"].min_value == config["mass"]["min_value"]
    assert sampler.samplers["mass"].max_value == config["mass"]["max_value"]
    assert config["mass"]["min_value"] <= cur_sample["mass"]
    assert config["mass"]["max_value"] >= cur_sample["mass"]

    # Check multirange_uniform sampler for gravity
    assert sampler.samplers["gravity"].intervals == config["gravity"]["intervals"]
    assert check_value_in_intervals(
        cur_sample["gravity"], sampler.samplers["gravity"].intervals
    )


def basic_tennis_sampler():
    return {"angle": {"sampler-type": "gaussian", "mean": 0, "var": 1}}


def test_tennis_sampler():
    config = basic_tennis_sampler()
    sampler = SamplerManager(config)
    assert sampler.is_empty() is False
    assert isinstance(sampler.samplers["angle"], GaussianSampler)

    cur_sample = sampler.sample_all()

    # Check angle gaussian sampler
    assert sampler.samplers["angle"].mean == config["angle"]["mean"]
    assert sampler.samplers["angle"].var == config["angle"]["var"]


def make_empty_sampler_config():
    return {}


def make_none_sampler_config():
    return None


def test_empty_samplers():
    empty_config = make_empty_sampler_config()
    empty_sampler = SamplerManager(empty_config)
    assert empty_sampler.is_empty()
    empty_cur_sample = empty_sampler.sample_all()
    assert empty_cur_sample == {}

    none_config = make_empty_sampler_config()
    none_sampler = SamplerManager(none_config)
    assert none_sampler.is_empty()
    none_cur_sample = none_sampler.sample_all()
    assert none_cur_sample == {}


def incorrect_uniform_sampler():
    # Do not specify required arguments to uniform sampler
    return {"mass": {"sampler-type": "uniform", "min-value": 10}}


def incorrect_sampler_config():
    # Do not specify 'sampler-type' key
    return {"mass": {"min-value": 2, "max-value": 30}}


def test_incorrect_uniform_sampler():
    config = incorrect_uniform_sampler()
    with pytest.raises(UnityException):
        SamplerManager(config)


def test_incorrect_sampler():
    config = incorrect_sampler_config()
    with pytest.raises(UnityException):
        SamplerManager(config)
