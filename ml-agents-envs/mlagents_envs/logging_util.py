import logging  # noqa I251
import sys

CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

_loggers = set()
_log_level = NOTSET
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEBUG_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
LOG_FORMAT = "[%(levelname)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name. The logger will use the log level
    specified by set_log_level()
    """
    logger = logging.getLogger(name=name)

    if _log_level == DEBUG:
        formatter = logging.Formatter(fmt=DEBUG_LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        formatter = logging.Formatter(fmt=LOG_FORMAT)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # If we've already set the log level, make sure new loggers use it
    if _log_level != NOTSET:
        logger.setLevel(_log_level)

    # Keep track of this logger so that we can change the log level later
    _loggers.add(logger)
    return logger


def set_log_level(log_level: int) -> None:
    """
    Set the ML-Agents logging level. This will also configure the logging format (if it hasn't already been set).
    """
    global _log_level
    _log_level = log_level

    for logger in _loggers:
        logger.setLevel(log_level)

    if log_level == DEBUG:
        formatter = logging.Formatter(fmt=DEBUG_LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        formatter = logging.Formatter(LOG_FORMAT)
    _set_formatter_for_all_loggers(formatter)


def _set_formatter_for_all_loggers(formatter: logging.Formatter) -> None:
    for logger in _loggers:
        for handler in logger.handlers[:]:
            handler.setFormatter(formatter)
