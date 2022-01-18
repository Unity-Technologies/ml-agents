# Unity ML-Agents Trainers

The `mlagents` Python package is part of the
[ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). `mlagents`
provides a set of reinforcement and imitation learning algorithms designed to be
used with Unity environments. The algorithms interface with the Python API
provided by the `mlagents_envs` package. See [here](../docs/Python-API.md) for
more information on `mlagents_envs`.

The algorithms can be accessed using the: `mlagents-learn` access point. See
[here](../docs/Training-ML-Agents.md) for more information on using this
package.

## Installation

Install the `mlagents` package with:

```sh
python -m pip install mlagents==0.28.0
```

## Usage & More Information

For more information on the ML-Agents Toolkit and how to instrument a Unity
scene with the ML-Agents SDK, check out the main
[ML-Agents Toolkit documentation](../docs/Readme.md).

## Limitations

- `mlagents` does not yet explicitly support multi-agent scenarios so training
  cooperative behavior among different agents is not stable.
- Resuming self-play from a checkpoint resets the reported ELO to the default
  value.
