import unittest.mock as mock
import pytest
import os
import tempfile

import mlagents.trainers.tensorflow_to_barracuda as tf2bc


def test_barracuda_converter():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    tmpfile = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()) + '.nn')

    # make sure file doesn't exist before conversion
    assert (not os.path.isfile(tmpfile))

    tf2bc.convert(path_prefix+'/BasicLearning.pb', tmpfile)

    # test if file exists after conversion
    assert (os.path.isfile(tmpfile))
    assert (os.path.getsize(tmpfile) > 100)

    # cleanup
    os.remove(tmpfile)
