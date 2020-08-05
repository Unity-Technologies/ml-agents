# # Unity ML-Agents Toolkit
import abc


class BaseSaver(abc.ABC):
    """This class is the base class for the Saver"""

    def __init__(self):
        """
        TBA
        """
        pass

    @abc.abstractmethod
    def register(self, module):
        pass

    @abc.abstractmethod
    def save_checkpoint(self, brain_name: str, step: int) -> str:
        pass

    @abc.abstractmethod
    def maybe_load(self):
        pass

    @abc.abstractmethod
    def export(self, output_filepath: str, brain_name: str) -> None:
        pass
