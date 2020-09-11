from typing import Optional

import os


def get_num_threads_to_use() -> Optional[int]:
    """
    Gets the number of threads to use. For most problems, 4 is all you
    need, but for smaller machines, we'd like to scale to less than that.
    By default, PyTorch uses 1/2 of the available cores.
    """
    num_cpus = _get_num_available_cpus()
    return max(min(num_cpus // 2, 4), 1) if num_cpus is not None else None


def _get_num_available_cpus() -> Optional[int]:
    """
    Returns number of CPUs using cgroups if possible. This accounts
    for Docker containers that are limited in cores.
    """
    period = _read_in_integer_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    quota = _read_in_integer_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    if period > 0 and quota > 0:
        return int(quota // period)
    else:
        return os.cpu_count()


def _read_in_integer_file(filename: str) -> int:
    try:
        with open(filename) as f:
            return int(f.read().rstrip())
    except FileNotFoundError:
        return -1
