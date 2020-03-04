import logging


def create_logger(name, log_level):
    date_format = "%Y-%m-%d %H:%M:%S"
    log_format = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)
    logger = logging.getLogger(name=name)
    return logger
