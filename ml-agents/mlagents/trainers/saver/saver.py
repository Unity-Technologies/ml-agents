# # Unity ML-Agents Toolkit
import abc


class Saver(abc.ABC):
    """This class is the base class for the Saver"""

    def __init__(self):
        """
        TBA
        """
        pass

    @abc.abstractmethod
    def register(self):
        pass

    @abc.abstractmethod
    def save_checkpoint(self):
        pass

    @abc.abstractmethod
    def maybe_load(self):
        pass

    @abc.abstractmethod
    def export(self):
        pass
