# Version of the library that will be used to upload to pypi
__version__ = "0.28.0.dev0"

# Git tag that will be checked to determine whether to trigger upload to pypi
__release_tag__ = None

try:
    from pettingzoo_unity.envs import UnityAECEnv  # noqa
    from pettingzoo_unity.envs import UnityParallelEnv  # noqa
    import pettingzoo_unity.envs  # noqa
except ImportError:
    pass
