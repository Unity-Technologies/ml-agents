from typing import Optional

import os


def get_num_threads_to_use() -> Optional[int]:
    """
    Gets the number of threads to use. For most problems, 4 is all you
    need, but for smaller machines, we'd like to scale to less than that.
    By default, PyTorch uses 1/2 of the available cores.
    """
    num_cpus = _get_num_cpus()
    return max(min(num_cpus // 2, 4), 1) if num_cpus is not None else None


def _get_num_cpus() -> Optional[int]:
    """
    Returns number of CPUs using cgroups if possible. This accounts
    for Docker containers that are limited in cores.
    """
    try:
        period = _read_in_integer_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        quota = _read_in_integer_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        num_cpus = quota // period
        if num_cpus > 0:
            return num_cpus
    except FileNotFoundError:
        pass
    return os.cpu_count()


def _read_in_integer_file(filename: str) -> int:
    with open(filename) as f:
        return int(f.readlines()[0])
