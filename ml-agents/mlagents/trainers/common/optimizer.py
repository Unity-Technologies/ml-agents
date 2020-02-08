import abc
from typing import Dict

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy import Policy


class Optimizer(abc.ABC):
    """
    Creates loss functions and auxillary networks (e.g. Q or Value) needed for training.
    Provides methods to update the Policy.
    """

    @abc.abstractmethod
    def __init__(self, policy: Policy):
        """
        Create loss functions and auxillary networks.
        :param policy: Policy object that is updated by the Optimizer
        """
        pass

    @abc.abstractmethod
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Update the Policy based on the batch that was passed in.
        :param batch: AgentBuffer that contains the minibatch of data used for this update.
        :param num_sequences: Number of recurrent sequences found in the minibatch.
        :return: A Dict containing statistics (name, value) from the update (e.g. loss)
        """
        pass
