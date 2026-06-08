# Unity ML-Agents PettingZoo Wrapper

With the increasing interest in multi-agent training with a gym-like API, we provide a PettingZoo Wrapper around the [Petting Zoo API](https://pettingzoo.farama.org/). Our wrapper provides interfaces on top of our `UnityEnvironment` class, which is the default way of interfacing with a Unity environment via Python.

## Installation and Examples

The PettingZoo wrapper is part of the `mlagents_envs` package. Please refer to the [mlagents_envs installation instructions](https://github.com/Unity-Technologies/ml-agents/blob/release/4.0.0/ml-agents-envs/README.md).

[[Colab] PettingZoo Wrapper Example](https://colab.research.google.com/github/Unity-Technologies/ml-agents/blob/develop-python-api-ga/ml-agents-envs/colabs/Colab_PettingZoo.ipynb)

This colab notebook demonstrates the example usage of the wrapper, including installation, basic usages, and an example with our [Striker vs Goalie environment](Learning-Environment-Examples.md#strikers-vs-goalie) which is a multi-agents environment with multiple different behavior names.

## API interface

This wrapper is compatible with PettingZoo API. Please check out [PettingZoo API page](https://pettingzoo.farama.org/api/aec/) for more details. Here's an example of interacting with wrapped environment:

```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs import UnityToPettingZooWrapper

unity_env = UnityEnvironment("StrikersVsGoalie")
env = UnityToPettingZooWrapper(unity_env)
env.reset()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    action = policy(observation, agent)
    env.step(action)
```

## Notes
- There is support for both [AEC](https://pettingzoo.farama.org/api/aec/) and [Parallel](https://pettingzoo.farama.org/api/parallel/) PettingZoo APIs.
- The AEC wrapper is compatible with PettingZoo (PZ) API interface but works in a slightly different way under the hood. For the AEC API, Instead of stepping the environment in every `env.step(action)`, the PZ wrapper will store the action, and will only perform environment stepping when all the agents requesting for actions in the current step have been assigned an action. This is for performance, considering that the communication between Unity and python is more efficient when data are sent in batches.
- Since the actions for the AEC wrapper are stored without applying them to the environment until all the actions are queued, some components of the API might behave in unexpected way. For example, a call to `env.reward` should return the instantaneous reward for that particular step, but the true reward would only be available when an actual environment step is performed. It's recommended that you follow the API definition for training (access rewards from `env.last()` instead of `env.reward`) and the underlying mechanism shouldn't affect training results.
- The environments will automatically reset when it's done, so `env.agent_iter(max_step)` will keep going on until the specified max step is reached (default: `2**63`). There is no need to call `env.reset()` except for the very beginning of instantiating an environment.
