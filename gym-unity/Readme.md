# Unity ML-Agents Gym Wrapper

A common way in which machine learning researchers interact with simulation environments is via a wrapper provided by OpenAI called `gym`. For more information on the gym interface, see [here](https://github.com/openai/gym). 

We provide a a gym wrapper, and instructions for using it with existing machine learning algorithms which utilize gyms. Both wrappers provide interfaces on top of our `UnityEnvironment` class, which is the default way of interfacing with a Unity environment via Python.

## Installation

The gym wrapper can be installed using:

```
pip install gym-unity
```

or by running the following from the `./gym-unity' directory of the repository:

```
pip install .
```


## Using the Gym Wrapper
The gym interface is available from `gym_unity.envs`. To launch an environmnent from the root of the project repository use:

```python
from gym_unity.envs import UnityEnv

env = UnityEnv(environment_filename, worker_id, default_visual, multiagent)
```

* `environment_filename` refers to the path to the Unity environment.
* `worker_id` refers to the port to use for communication with the environment.
* `default_visual` refers to whether to use visual observations (True) or vector observations (False) as the default observation provided by the `reset` and `step` functions.
* `multiagent` refers to whether you intent to launch an environment which contains more than one agent.

The returned environment `env` will function as a gym.

For more on using the gym interface, see our [Jupyter Notebook tutorial](../python/notebooks/getting-started-gym.ipynb).

## Limitation

 * It is only possible to use an environment with a single Brain.
 * By default the first visual observation is provided as the `observation`, if present. Otherwise vector observations are provided. 
 * All `BrainInfo` output from the environment can still be accessed from the `info` provided by `env.step(action)`.
 * Environment registration for use with `gym.make()` is currently not supported.