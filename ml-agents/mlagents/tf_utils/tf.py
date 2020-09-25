# This should be the only place that we import tensorflow directly.
# Everywhere else is caught by the banned-modules setting for flake8

from distutils.version import LooseVersion

try:
    import tensorflow as tf  # noqa I201

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
except ImportError:
    tf = None


def is_available():
    """
    Returns whether Torch is available in this Python environment
    """
    return tf is not None


def set_warnings_enabled(is_enabled: bool) -> None:
    """
    Enable or disable tensorflow warnings (notably, this disables deprecation warnings.
    :param is_enabled:
    """
    if is_available():
        level = tf_logging.WARN if is_enabled else tf_logging.ERROR
        tf_logging.set_verbosity(level)


def generate_session_config() -> "tf.ConfigProto":
    """
    Generate a ConfigProto to use for ML-Agents that doesn't consume all of the GPU memory
    and allows for soft placement in the case of multi-GPU.
    """
    if is_available():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # For multi-GPU training, set allow_soft_placement to True to allow
        # placing the operation into an alternative device automatically
        # to prevent from exceptions if the device doesn't suppport the operation
        # or the device does not exist
        config.allow_soft_placement = True
        return config
    else:
        return None
