# Version of the library that will be used to upload to pypi
__version__ = "0.28.0.dev0"

# Git tag that will be checked to determine whether to trigger upload to pypi
__release_tag__ = None

try:
    import gym_unity.envs  # noqa
except ImportError:
    # Try here because when calling setup, we access __version__ but do not have
    # gym installed yet. This is to make installation not raise an error.
    # we want to be able to use the gym registry after calling `import gym_unity`
    pass
