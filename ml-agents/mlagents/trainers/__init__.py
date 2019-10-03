import logging

def warnings_as_errors(log_record):
    # Raise deprecated warnings as exceptions.
    if log_record.levelno == logging.WARNING and "deprecated" in log_record.msg:
        merged = log_record.getMessage()
        raise RuntimeError(merged)
    return True

# TODO only enable this with a environment variable
if False:
    logging.getLogger('tensorflow').addFilter(warnings_as_errors)


# TODO better place to put this? move everything to tf.py?
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
tf.disable_v2_behavior()
