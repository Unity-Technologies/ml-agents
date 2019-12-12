"""
Contains exceptions for the trainers package.
"""


class TrainerError(Exception):
    """
    Any error related to the trainers in the ML-Agents Toolkit.
    """

    pass


class TrainerConfigError(Exception):
    """
    Any error related to the configuration of trainers in the ML-Agents Toolkit.
    """

    pass


class CurriculumError(TrainerError):
    """
    Any error related to training with a curriculum.
    """

    pass


class CurriculumLoadingError(CurriculumError):
    """
    Any error related to loading the Curriculum config file.
    """

    pass


class CurriculumConfigError(CurriculumError):
    """
    Any error related to processing the Curriculum config file.
    """

    pass


class MetaCurriculumError(TrainerError):
    """
    Any error related to the configuration of a metacurriculum.
    """

    pass


class SamplerException(TrainerError):
    """
    Related to errors with the sampler actions.
    """

    pass
