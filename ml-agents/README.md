# Unity ML-Agents Python Interface and Trainers

The `mlagents` Python package is part of the
[ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).
`mlagents` provides a Python API that allows direct interaction with the Unity
game engine as well as a collection of trainers and algorithms to train agents
in Unity environments.

The `mlagents` Python package contains two sub packages:

* `mlagents.envs`: A low level API which allows you to interact directly with a
  Unity Environment. See
  [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md)
  for more information on using this package.

* `mlagents.trainers`: A set of Reinforcement Learning algorithms designed to be
  used with Unity environments. Access them using the: `mlagents-learn` access
  point. See
  [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-ML-Agents.md)
  for more information on using this package.

## Installation

First install `ml-agents-envs` per instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/ml-agents-envs/README.md) and then install the `mlagents` package:

```sh
cd mlagents
pip install mlagents
```
**NOTE** Please install as per above instructions and not from pypi.

## Usage & More Information

For more detailed documentation, check out the
[ML-Agents Toolkit documentation.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md)
