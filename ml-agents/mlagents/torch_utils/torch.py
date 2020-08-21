try:
    import torch

    # Known PyLint compatibility with PyTorch https://github.com/pytorch/pytorch/issues/701
    # pylint: disable=E1101
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    # pylint: disable=E1101
except ImportError:
    torch = None


def is_available():
    """
    Returns whether Torch is available in this Python environment
    """
    return torch is not None
