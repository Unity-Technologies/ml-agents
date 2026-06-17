import pytest
from unittest import mock

import torch  # noqa I201

from mlagents.torch_utils import set_torch_config, default_device
from mlagents.trainers.settings import TorchSettings


@pytest.mark.parametrize(
    "device_str, expected_type, expected_index, expected_tensor_type",
    [
        ("cpu", "cpu", None, torch.float32),
        ("cuda", "cuda", None, torch.float32),
        ("cuda:42", "cuda", 42, torch.float32),
        ("opengl", "opengl", None, torch.float32),
    ],
)
@mock.patch.object(torch, "set_default_dtype")
@mock.patch.object(torch, "set_default_device")
def test_set_torch_device(
    mock_set_default_device,
    mock_set_default_tensor_type,
    device_str,
    expected_type,
    expected_index,
    expected_tensor_type,
):
    try:
        torch_settings = TorchSettings(device=device_str)
        set_torch_config(torch_settings)
        assert default_device().type == expected_type
        if expected_index is None:
            assert default_device().index is None
        else:
            assert default_device().index == expected_index
        mock_set_default_tensor_type.assert_called_once_with(expected_tensor_type)
    except Exception:
        raise
    finally:
        # restore the defaults
        torch_settings = TorchSettings(device=None)
        set_torch_config(torch_settings)


@pytest.mark.parametrize(
    "device_str, expect_set_default_device",
    [
        # GPU-like devices must set the *full* device as default, preserving the
        # index (regression for cuda:1 landing tensors on cuda:0).
        ("cuda", True),
        ("cuda:1", True),
        ("xpu", True),
        ("mps", True),
        # cpu (and unknown/non-GPU types) must not call set_default_device.
        ("cpu", False),
        ("opengl", False),
    ],
)
@mock.patch.object(torch, "set_default_dtype")
@mock.patch.object(torch, "set_default_device")
def test_set_torch_config_default_device(
    mock_set_default_device,
    mock_set_default_dtype,
    device_str,
    expect_set_default_device,
):
    try:
        set_torch_config(TorchSettings(device=device_str))
        if expect_set_default_device:
            # The full device (including any index) is passed, not just the type.
            mock_set_default_device.assert_called_once_with(torch.device(device_str))
        else:
            mock_set_default_device.assert_not_called()
        # dtype is always pinned to float32 regardless of device.
        mock_set_default_dtype.assert_called_once_with(torch.float32)
    finally:
        # restore the defaults
        set_torch_config(TorchSettings(device=None))
