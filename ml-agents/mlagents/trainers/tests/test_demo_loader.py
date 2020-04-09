import io
import os
from unittest import mock
import numpy as np
import pytest
import tempfile

from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)

from mlagents.trainers.demo_loader import (
    load_demonstration,
    demo_to_buffer,
    get_demo_files,
    write_delimited,
)


def test_load_demo():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    behavior_spec, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test.demo"
    )
    assert np.sum(behavior_spec.observation_shapes[0]) == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test.demo", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1


def test_load_demo_dir():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    behavior_spec, pair_infos, total_expected = load_demonstration(
        path_prefix + "/test_demo_dir"
    )
    assert np.sum(behavior_spec.observation_shapes[0]) == 8
    assert len(pair_infos) == total_expected

    _, demo_buffer = demo_to_buffer(path_prefix + "/test_demo_dir", 1)
    assert len(demo_buffer["actions"]) == total_expected - 1


def test_edge_cases():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
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
        invalid_fname = os.path.join(tmpdirname, "mydemo.notademo")
        with open(invalid_fname, "w") as f:
            f.write("I'm not a demo")
        with pytest.raises(ValueError):
            get_demo_files(invalid_fname)
        # invalid directory
        with pytest.raises(ValueError):
            get_demo_files(tmpdirname)
        # valid file
        valid_fname = os.path.join(tmpdirname, "mydemo.demo")
        with open(valid_fname, "w") as f:
            f.write("I'm a demo file")
        assert get_demo_files(valid_fname) == [valid_fname]
        # valid directory
        assert get_demo_files(tmpdirname) == [valid_fname]


@mock.patch("mlagents.trainers.demo_loader.get_demo_files", return_value=["foo.demo"])
def test_unsupported_version_raises_error(mock_get_demo_files):
    # Create a metadata proto with an unsupported version
    bad_metadata = DemonstrationMetaProto()
    bad_metadata.api_version = 1337

    # Write the metadata to a temporary buffer, which will get returned by open()
    buffer = io.BytesIO()
    write_delimited(buffer, bad_metadata)
    m = mock.mock_open(read_data=buffer.getvalue())

    # Make sure that we get a RuntimeError when trying to load this.
    with mock.patch("builtins.open", m):
        with pytest.raises(RuntimeError):
            load_demonstration("foo")
