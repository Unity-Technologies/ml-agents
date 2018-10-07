"""
Contains exceptions for the trainers package.
"""

class TrainerError(Exception):
    """
    Any error related to the trainers in the ML-Agents Toolkit.
    """
    pass

class CurriculumError(TrainerError):
    """
    Any error related to training with a curriculum.
    """
    pass

class MetaCurriculumError(TrainerError):
    """
    Any error related to the configuration of a metacurriculum.
    """
