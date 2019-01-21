import unittest.mock as mock
import pytest
import os
import tempfile

from mlagents.trainers.tensorflow_to_barracuda import ModelConverter



def test_barracuda_converter():
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    tmpfile = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()) + '.nn')

    # make sure file doesn't exist before conversion
    assert (not os.path.isfile(tmpfile))

    mc = ModelConverter()
    mc.process(path_prefix+'/test.demo', tmpfile)

    # test if file exists after conversion
    assert (os.path.isfile(tmpfile))

    # cleanup
    os.remove(tmpfile)