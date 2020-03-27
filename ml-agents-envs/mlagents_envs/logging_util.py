import logging  # noqa I251

CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

_loggers = set()
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"


def get_logger(name):
    logger = logging.getLogger(name=name)
    _loggers.add(logger)
    return logger


def set_log_level(log_level: int) -> None:
    logging.basicConfig(level=log_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    for logger in _loggers:
        logger.setLevel(log_level)
