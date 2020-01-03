import os
import pytest

from mlagents.trainers.demo_loader import (
    load_demonstration,
    demo_to_buffer,
    get_demo_files,
)


def test_load_demo():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    brain_parameters, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test.demo"
    )
    assert brain_parameters.brain_name == "Ball3DBrain"
    assert brain_parameters.vector_observation_space_size == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test.demo", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1


def test_load_demo_dir():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    brain_parameters, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test_demo_dir"
    )
    assert brain_parameters.brain_name == "3DBall"
    assert brain_parameters.vector_observation_space_size == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test_demo_dir", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1


def test_nonexisting_paths():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(FileNotFoundError):
        get_demo_files(os.path.join(path_prefix, "idontexistfile.demo"))
    with pytest.raises(FileNotFoundError):
        get_demo_files(os.path.join(path_prefix, "idontexistdirectory"))
