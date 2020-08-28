import abc
from typing import Dict

from mlagents.trainers.buffer import AgentBuffer


class Optimizer(abc.ABC):
    """
    Creates loss functions and auxillary networks (e.g. Q or Value) needed for training.
    Provides methods to update the Policy.
    """

    def __init__(self):
        self.reward_signals = {}

    @abc.abstractmethod
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Update the Policy based on the batch that was passed in.
        :param batch: AgentBuffer that contains the minibatch of data used for this update.
        :param num_sequences: Number of recurrent sequences found in the minibatch.
        :return: A Dict containing statistics (name, value) from the update (e.g. loss)
        """
        pass
