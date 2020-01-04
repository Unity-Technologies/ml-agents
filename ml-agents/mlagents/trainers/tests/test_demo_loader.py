import os
import pytest
import tempfile

from mlagents.trainers.demo_loader import (
    load_demonstration,
    demo_to_buffer,
    get_demo_files,
)

path_prefix = os.path.dirname(os.path.abspath(__file__))


def test_load_demo():
    brain_parameters, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test.demo"
    )
    assert brain_parameters.brain_name == "Ball3DBrain"
    assert brain_parameters.vector_observation_space_size == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test.demo", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1


def test_load_demo_dir():
    brain_parameters, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test_demo_dir"
    )
    assert brain_parameters.brain_name == "3DBall"
    assert brain_parameters.vector_observation_space_size == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test_demo_dir", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1


def test_edge_cases():
    # nonexistent file and directory
    with pytest.raises(FileNotFoundError):
        get_demo_files(os.path.join(path_prefix, "nonexistent_file.demo"))
    with pytest.raises(FileNotFoundError):
        get_demo_files(os.path.join(path_prefix, "nonexistent_directory"))
    with tempfile.TemporaryDirectory() as tmpdirname:
        # empty directory
        with pytest.raises(ValueError):
            get_demo_files(tmpdirname)
        # invalid file
        invalid_fname = tmpdirname + "/mydemo.notademo"
        with open(invalid_fname, "w") as f:
            f.write("I'm not a demo")
        with pytest.raises(ValueError):
            get_demo_files(invalid_fname)
        # valid file
        valid_fname = tmpdirname + "/mydemo.demo"
        with open(valid_fname, "w") as f:
            f.write("I'm a demo file")
        assert get_demo_files(valid_fname) == [valid_fname]
        # valid directory
        assert get_demo_files(tmpdirname) == [valid_fname]
