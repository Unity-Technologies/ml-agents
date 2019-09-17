import logging

logger = logging.getLogger("mlagents.envs")


class UnityException(Exception):
    """
    Any error related to ml-agents environment.
    """

    pass


class UnityEnvironmentException(UnityException):
    """
    Related to errors starting and closing environment.
    """

    pass


class UnityCommunicationException(UnityException):
    """
    Related to errors with the communicator.
    """

    pass


class UnityActionException(UnityException):
    """
    Related to errors with sending actions.
    """

    pass


class SamplerException(UnityException):
    """
    Related to errors with the sampler actions.
    """

    pass


class UnityTimeOutException(UnityException):
    """
    Related to errors with communication timeouts.
    """

    pass


class UnityWorkerInUseException(UnityException):
    """
    This error occurs when the port for a certain worker ID is already reserved.
    """

    MESSAGE_TEMPLATE = (
        "Couldn't start socket communication because worker number {} is still in use. "
        "You may need to manually close a previously opened environment "
        "or use a different worker number."
    )

    def __init__(self, worker_id):
        message = self.MESSAGE_TEMPLATE.format(str(worker_id))
        super(UnityWorkerInUseException, self).__init__(message)
