import logging


def warnings_as_errors(log_record):
    # Raise deprecated warnings as exceptions.
    if log_record.levelno == logging.WARNING and "deprecated" in log_record.msg:
        merged = log_record.getMessage()
        raise RuntimeError(merged)
    return True


# TODO only enable this with a environment variable
if False:
    logging.getLogger("tensorflow").addFilter(warnings_as_errors)

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

is_tf2 = True
if is_tf2:
    tf.disable_v2_behavior()
