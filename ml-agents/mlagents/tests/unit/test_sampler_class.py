import pytest

from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.sampler_class import (
    UniformSampler,
    MultiRangeUniformSampler,
    GaussianSampler,
)
from mlagents.trainers.exception import TrainerError


def sampler_config_1():
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


def test_sampler_config_1():
    config = sampler_config_1()
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


def sampler_config_2():
    return {"angle": {"sampler-type": "gaussian", "mean": 0, "st_dev": 1}}


def test_sampler_config_2():
    config = sampler_config_2()
    sampler = SamplerManager(config)
    assert sampler.is_empty() is False
    assert isinstance(sampler.samplers["angle"], GaussianSampler)

    # Check angle gaussian sampler
    assert sampler.samplers["angle"].mean == config["angle"]["mean"]
    assert sampler.samplers["angle"].st_dev == config["angle"]["st_dev"]


def test_empty_samplers():
    empty_sampler = SamplerManager({})
    assert empty_sampler.is_empty()
    empty_cur_sample = empty_sampler.sample_all()
    assert empty_cur_sample == {}

    none_sampler = SamplerManager(None)
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
    with pytest.raises(TrainerError):
        SamplerManager(config)


def test_incorrect_sampler():
    config = incorrect_sampler_config()
    with pytest.raises(TrainerError):
        SamplerManager(config)
