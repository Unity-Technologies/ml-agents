import logging
import horovod.tensorflow as hvd


def create_logger(name, log_level):
    date_format = "%Y-%m-%d %H:%M:%S"
    horovod_worker = "worker-%s " % str(hvd.rank())
    log_format = horovod_worker + "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)
    logger = logging.getLogger(name=name)
    return logger
