try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


def get_hvd():
    return hvd


def get_rank():
    rank = hvd.rank() if hvd else None
    return rank
