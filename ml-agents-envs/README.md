# Unity ML-Agents Python Interface

The `mlagents_envs` Python package is part of the
[ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).
`mlagents_envs` provides a Python API that allows direct interaction with the Unity
game engine. It is used by the trainer implementation in `mlagents` as well as
the `gym-unity` package to perform reinforcement learning within Unity. `mlagents_envs` can be
used independently of `mlagents` for Python communication.

The `mlagents_envs` Python package contains a low level API which allows you to interact
directly with a Unity Environment. See [here](../docs/Python-API.md) for more information
on using this API.

## Installation

Install the `mlagents_envs` package with:

```sh
pip install mlagents_envs
```

## Usage & More Information

For more detailed documentation, check out the
[ML-Agents Toolkit documentation.](../docs/Readme.md)

## Limitations
 - `mlagents_envs` uses localhost ports to exchange data between Unity and Python. As such,
 multiple instances can have their ports collide leading to errors. Make sure to use a
 different port if you are using multiple instances of `UnityEnvironment'.
 - Communication between Unity and the Python `UnityEnvironment` is not secure.
 - On Linux, ports are not released immediately after the communication closes. As such, you
 cannot reuse ports right after closing a `UnityEnvironment`.
