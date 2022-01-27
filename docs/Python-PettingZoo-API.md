# Unity ML-Agents PettingZoo Wrapper

With the increasing interest in multi-agent training with a gym-like API, we provide a
PettingZoo Wrapper around the [Petting Zoo API](https://www.pettingzoo.ml/). Our wrapper
provides interfaces on top of our `UnityEnvironment` class, which is the default way of
interfacing with a Unity environment via Python.

## Installation and Examples

[[Colab] PettingZoo Wrapper Example](https://colab.research.google.com/github/Unity-Technologies/ml-agents/blob/develop-python-api-ga/ml-agents-envs/colabs/Colab_PettingZoo.ipynb)

This colab notebook demonstrates the example usage of the wrapper, including installation,
basic usages, and an example with our
[Striker vs Goalie environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#strikers-vs-goalie)
which is a multi-agents environment with multiple different behavior names.

## API interface

This wrapper is compatible with PettingZoo API. Please check out
[PettingZoo API page](https://www.pettingzoo.ml/api) for more details.
Here's an example of interacting with wrapped environment:

```python
unity_env = UnityEnvironment("StrikersVsGoalie")
env = UnityToPettingZooWrapper(unity_env)
env.reset()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    action = policy(observation, agent)
    env.step(action)
```

## Notes
- The wrapper is compatible with PettingZoo API interface but works in a little bit
  different way under the hood. Instead of stepping the environment in every `env.step(action)`,
  our environment will store the action, and will only perform environment stepping when all the
  agents requesting for actions in the current step have been assigned an action. This is for
  performance consideration that the communication between Unity and python is more efficient
  when data are sent in batches.
- Since the actions are stored with no environment, some additional API might behave in unexpected
  way. Specifically, `env.reward` should return the instant reward in that particular step, but you
  would only see those reward when an actual environment step is performed. It's recommended that
  you follow the API definition for training (access rewards from `env.last()` instead of
  `env.reward`) and the underlying mechanism shouldn't affect training results.
- Advanced features in PettingZoo like Parallel API is not guaranteed to work with this wrapper.
- The environments will automatically reset when it's done, so `env.agent_iter(max_step)` will
  keep going on until the specified max step is reached (default: `2**63`). There is no need to
  call `env.reset()` except for the very beginning of instantiating an environment.

