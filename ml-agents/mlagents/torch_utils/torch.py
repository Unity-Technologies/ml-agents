import os

from mlagents.torch_utils import cpu_utils

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
