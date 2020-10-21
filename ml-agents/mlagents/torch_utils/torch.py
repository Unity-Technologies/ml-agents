import os

from distutils.version import LooseVersion
import pkg_resources
from mlagents.torch_utils import cpu_utils


def assert_torch_installed():
    # Check that torch version 1.6.0 or later has been installed. If not, refer
    # user to the PyTorch webpage for install instructions.
    torch_pkg = None
    try:
        torch_pkg = pkg_resources.get_distribution("torch")
    except pkg_resources.DistributionNotFound:
        pass
    assert torch_pkg is not None and LooseVersion(torch_pkg.version) >= LooseVersion(
        "1.6.0"
    ), (
        "A compatible version of PyTorch was not installed. Please visit the PyTorch homepage "
        + "(https://pytorch.org/get-started/locally/) and follow the instructions to install. "
        + "Version 1.6.0 and later are supported."
    )


assert_torch_installed()

# This should be the only place that we import torch directly.
# Everywhere else is caught by the banned-modules setting for flake8
import torch  # noqa I201


torch.set_num_threads(cpu_utils.get_num_threads_to_use())
os.environ["KMP_BLOCKTIME"] = "0"

# Known PyLint compatibility with PyTorch https://github.com/pytorch/pytorch/issues/701
# pylint: disable=E1101
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")
nn = torch.nn


def default_device():
    return device
