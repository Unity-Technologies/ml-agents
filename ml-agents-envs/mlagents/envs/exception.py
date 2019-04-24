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


class UnityActionException(UnityException):
    """
    Related to errors with sending actions.
    """

    pass


class UnityTimeOutException(UnityException):
    """
    Related to errors with communication timeouts.
    """

    def __init__(self, message, log_file_path=None):
        if log_file_path is not None:
            try:
                with open(log_file_path, "r") as f:
                    printing = False
                    unity_error = "\n"
                    for l in f:
                        l = l.strip()
                        if (l == "Exception") or (l == "Error"):
                            printing = True
                            unity_error += "----------------------\n"
                        if l == "":
                            printing = False
                        if printing:
                            unity_error += l + "\n"
                    logger.info(unity_error)
                    logger.error(
                        "An error might have occured in the environment. "
                        "You can check the logfile for more information at {}".format(
                            log_file_path
                        )
                    )
            except:
                logger.error(
                    "An error might have occured in the environment. "
                    "No UnitySDK.log file could be found."
                )
        super(UnityTimeOutException, self).__init__(message)


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
