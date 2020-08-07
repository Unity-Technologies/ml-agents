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


class UnityCommunicatorStoppedException(UnityException):
    """
    Raised when communicator has stopped gracefully.
    """

    pass


class UnityObservationException(UnityException):
    """
    Related to errors with receiving observations.
    """

    pass


class UnityActionException(UnityException):
    """
    Related to errors with sending actions.
    """

    pass


class UnityTimeOutException(UnityException):
    """
    Related to errors with communication timeouts.
    """

    pass


class UnitySideChannelException(UnityException):
    """
    Related to errors with side channels.
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
        super().__init__(message)


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass
