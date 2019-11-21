# This should be the only place that we import tensorflow directly.
# Everywhere else is caught by the banned-modules setting for flake8
import tensorflow as tf  # noqa I201
from distutils.version import LooseVersion


# LooseVersion handles things "1.2.3a" or "4.5.6-rc7" fairly sensibly.
_is_tensorflow2 = LooseVersion(tf.__version__) >= LooseVersion("2.0.0")

if _is_tensorflow2:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf_logging = tf.logging
else:
    try:
        # Newer versions of tf 1.x will complain that tf.logging is deprecated
        tf_logging = tf.compat.v1.logging
    except AttributeError:
        # Fall back to the safe import, even if it might generate a warning or two.
        tf_logging = tf.logging


def set_warnings_enabled(is_enabled: bool) -> None:
    """
    Enable or disable tensorflow warnings (notabley, this disables deprecation warnings.
    :param is_enabled:
    """
    level = tf_logging.WARN if is_enabled else tf_logging.ERROR
    tf_logging.set_verbosity(level)


# HACK numpy here for now
import numpy as np
import traceback
# TODO condition on env variable
__old_np_array = np.array
__old_np_zeros = np.zeros

def np_array_no_float64(*args, **kwargs):
    res = __old_np_array(*args, **kwargs)
    if res.dtype == np.float64:
        tb = traceback.extract_stack()
        # last entry, tb[-1], in the stack is this file.
        # we want the calling function, so use tb[-2]
        filename = tb[-2].filename
        # HACK - only raise if this came from mlagents code, not tensorflow
        if "ml-agents/mlagents" in filename and "tensorflow_to_barracuda.py" not in filename:
            #and "ml-agents/mlagents/trainers/tests" not in filename\
            raise ValueError(f"dtype={kwargs.get('dtype')}")
    return res
np.array = np_array_no_float64

