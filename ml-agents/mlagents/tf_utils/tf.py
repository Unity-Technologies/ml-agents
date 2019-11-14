# This should be the only place that we import tensorflow directly.
# Everywhere else is caught by the banned-modules setting for flake8
import tensorflow as tf  # noqa I201
from distutils.version import LooseVersion


# LooseVersion handles things "1.2.3a" or "4.5.6-rc7" fairly sensibly.
_is_tensorflow2 = LooseVersion(tf.__version__) >= LooseVersion("2.0.0")

if _is_tensorflow2:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    pass
