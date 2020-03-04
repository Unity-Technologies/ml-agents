import logging


def create_logger(name):
    log_level = logging.INFO
    log_format = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name=name)
    return logger


# TODO
"""
1. change mlagents.trainers to mlagents_trainers
2. logger should be in lowercase in all files
3. if logger is not used in a file then it should be deleted
"""
