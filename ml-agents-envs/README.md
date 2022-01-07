# Unity ML-Agents Python Interface

The `mlagents_envs` Python package is part of the
[ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).
`mlagents_envs` provides a Python API that allows direct interaction with the
Unity game engine. It is used by the trainer implementation in `mlagents` as
well as the `gym-unity` package to perform reinforcement learning within Unity.
`mlagents_envs` can be used independently of `mlagents` for Python
communication.

## Installation

Install the `mlagents_envs` package with:

```sh
python -m pip install mlagents_envs==0.28.0
```

## Usage & More Information

See the [Python API Guide](../docs/Python-API.md) for more information on how to
use the API to interact with a Unity environment.

For more information on the ML-Agents Toolkit and how to instrument a Unity
scene with the ML-Agents SDK, check out the main
[ML-Agents Toolkit documentation](../docs/Readme.md).

## Limitations

- `mlagents_envs` uses localhost ports to exchange data between Unity and
  Python. As such, multiple instances can have their ports collide, leading to
  errors. Make sure to use a different port if you are using multiple instances
  of `UnityEnvironment`.
- Communication between Unity and the Python `UnityEnvironment` is not secure.
- On Linux, ports are not released immediately after the communication closes.
  As such, you cannot reuse ports right after closing a `UnityEnvironment`.
