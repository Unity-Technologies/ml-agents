import os
import numpy as np

from mlagents.trainers.demo_loader import load_demonstration, demo_to_buffer


def test_load_demo():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    group_spec, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test.demo"
    )
    assert np.sum(group_spec.observation_shapes[0]) == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test.demo", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1


def test_load_demo_dir():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    group_spec, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test_demo_dir"
    )
    assert np.sum(group_spec.observation_shapes[0]) == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test_demo_dir", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1
