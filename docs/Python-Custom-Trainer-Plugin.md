# Unity Ml-Agents Custom trainers Plugin

As an attempt to bring a wider variety of reinforcement learning algorithms to our users, we have added custom trainers
capabilities. we introduce an extensible plugin system to define new trainers based on the High level trainer API
in `Ml-agents` Package. This will allow rerouting `mlagents-learn` CLI to custom trainers and extending the config files
with hyper-parameters specific to your new trainers. We will expose a high-level extensible trainer (both on-policy,
and off-policy trainers) optimizer and hyperparameter classes with documentation for the use of this plugin. For more
infromation on how python plugin system works see [Plugin interfaces](Training-Plugins.md).
## Overview
Model-free RL algorithms generally fall into two broad categories: on-policy and off-policy. On-policy algorithms perform updates based on data gathered from the current policy. Off-policy algorithms learn a Q function from a buffer of previous data, then use this Q function to make decisions. Off-policy algorithms have three key benefits in the context of ML-Agents: They tend to use fewer samples than on-policy as they can pull and re-use data from the buffer many times. They allow player demonstrations to be inserted in-line with RL data into the buffer, enabling new ways of doing imitation learning by streaming player data.

To add new custom trainers to ML-agents, you would need to create a new python package.
To give you an idea of how to structure your package, we have created a [mlagents_trainer_plugin](../ml-agents-trainer-plugin) package ourselves as an
example, with implementation of `A2c` and `DQN` algorithms. You would need a `setup.py` file to list extra requirements and
register the new RL algorithm in ml-agents ecosystem and be able to call `mlagents-learn` CLI with your customized
configuration.


```shell
├── mlagents_trainer_plugin
│    ├── __init__.py
│    ├── a2c
│    │    ├── __init__.py
│    │    ├── a2c_3DBall.yaml
│    │    ├── a2c_optimizer.py
│    │    └── a2c_trainer.py
│    └── dqn
│        ├── __init__.py
│        ├── dqn_basic.yaml
│        ├── dqn_optimizer.py
│        └── dqn_trainer.py
└── setup.py
```
## Installation and Execution
If you haven't already, follow the [installation instructions](Installation.md). Once you have the `ml-agents-env` and `ml-agents` packages you can install the plugin package. From the repository's root directory install `ml-agents-trainer-plugin` (or replace with the name of your plugin folder).

```sh
pip3 install -e <./ml-agents-trainer-plugin>
```

Following the previous installations your package is added as an entrypoint and you can use a config file with new
trainers:
```sh
mlagents-learn ml-agents-trainer-plugin/mlagents_trainer_plugin/a2c/a2c_3DBall.yaml --run-id <run-id-name>
--env <env-executable>
```

## Tutorial
Here’s a step-by-step [tutorial](Tutorial-Custom-Trainer-Plugin.md) on how to write a setup file and extend ml-agents trainers, optimizers, and
hyperparameter settings.To extend ML-agents classes see references on
[trainers](Python-On-Off-Policy-Trainer-Documentation.md) and [Optimizer](Python-Optimizer-Documentation.md).
