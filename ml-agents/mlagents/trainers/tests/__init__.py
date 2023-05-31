import os

# Opt-in checking mode to ensure that we always create numpy arrays using float32
if os.getenv("TEST_ENFORCE_NUMPY_FLOAT32"):
    # This file is importer by pytest multiple times, but this breaks the patching
    # Removing the env variable seems the easiest way to prevent this.
    del os.environ["TEST_ENFORCE_NUMPY_FLOAT32"]
    import numpy as np
    import traceback

    __old_np_array = np.array
    __old_np_zeros = np.zeros
    __old_np_ones = np.ones

    def _check_no_float64(arr, kwargs_dtype):
        if arr.dtype == np.float64:
            tb = traceback.extract_stack()
            # tb[-1] in the stack is this function.
            # tb[-2] is the wrapper function, e.g. np_array_no_float64
            # we want the calling function, so use tb[-3]
            filename = tb[-3].filename
            # Only raise if this came from mlagents code
            if (
                "ml-agents/mlagents" in filename
                or "ml-agents-envs/mlagents" in filename
            ):
                raise ValueError(
                    f"float64 array created. Set dtype=np.float32 instead of current dtype={kwargs_dtype}. "
                    f"Run pytest with TEST_ENFORCE_NUMPY_FLOAT32=1 to confirm fix."
                )

    def np_array_no_float64(*args, **kwargs):
        res = __old_np_array(*args, **kwargs)
        _check_no_float64(res, kwargs.get("dtype"))
        return res

    def np_zeros_no_float64(*args, **kwargs):
        res = __old_np_zeros(*args, **kwargs)
        _check_no_float64(res, kwargs.get("dtype"))
        return res

    def np_ones_no_float64(*args, **kwargs):
        res = __old_np_ones(*args, **kwargs)
        _check_no_float64(res, kwargs.get("dtype"))
        return res

    np.array = np_array_no_float64
    np.zeros = np_zeros_no_float64
    np.ones = np_ones_no_float64


if os.getenv("TEST_ENFORCE_BUFFER_KEY_TYPES"):
    from mlagents.trainers.buffer import AgentBuffer

    AgentBuffer.CHECK_KEY_TYPES_AT_RUNTIME = True
