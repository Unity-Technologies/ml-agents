from typing import Set
import argparse


class DetectDefault(argparse.Action):
    """
    Internal custom Action to help detect arguments that aren't default.
    """

    non_default_args: Set[str] = set()

    def __call__(self, arg_parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        DetectDefault.non_default_args.add(self.dest)


class DetectDefaultStoreTrue(DetectDefault):
    """
    Internal class to help detect arguments that aren't default.
    Used for store_true arguments.
    """

    def __init__(self, nargs=0, **kwargs):
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, arg_parser, namespace, values, option_string=None):
        super().__call__(arg_parser, namespace, True, option_string)


class StoreConfigFile(argparse.Action):
    """
    Custom Action to store the config file location not as part of the CLI args.
    This is because we want to maintain an equivalence between the config file's
    contents and the args themselves.
    """

    trainer_config_path: str

    def __call__(self, arg_parser, namespace, values, option_string=None):
        delattr(namespace, self.dest)
        StoreConfigFile.trainer_config_path = values
