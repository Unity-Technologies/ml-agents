# This should be the only place that we import tensorflow directly.
# Everywhere else is caught by the banned-modules setting for flake8
import tensorflow as tf  # noqa I201
from distutils.version import LooseVersion


# LooseVersion handles things "1.2.3a" or "4.5.6-rc7" fairly sensibly.
_is_tensorflow2 = LooseVersion(tf.__version__) >= LooseVersion("2.0.0")

# A few things that we use live in different places between tensorflow 1.x and 2.x
# If anything new is added, please add it here

if _is_tensorflow2:
    import tensorflow.compat.v1 as tf

    tf_variance_scaling = tf.initializers.variance_scaling
    tf_flatten = tf.layers.flatten
    tf_rnn = tf.nn.rnn_cell

    tf.disable_v2_behavior()
else:
    import tensorflow.contrib.layers as c_layers

    tf_variance_scaling = c_layers.variance_scaling_initializer
    tf_flatten = c_layers.flatten
    tf_rnn = tf.contrib.rnn
