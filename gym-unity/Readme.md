# Unity ML-Agents Gym Interface

A common way in which machine learning researchers interact with simulation environments is via a wrapper provided by OpenAI called `gym`. For more information on the gym interface, see [here](https://github.com/openai/gym). 

We provide a two gym wrappers, and instructions for using them with existing research projects which utilize gyms. 

## Installation

The gym wrapper can be installed using:

```
pip install gym-unity
```

or by running the following from the root directory of the repository:

```
pip install -e ./gym-unity
```


## Using the Gym Interface
The gym interface is available from `gym_unity.envs`. To launch a single agent environmnent from the root of the project repository use:

```python
from gym_unity.envs import UnityEnv

env = UnityEnv(environment_filename, worker_id, default_visual)
```

To launch a multi-agent environment, use:

```python
from gym_unity.envs import UnityMultiAgentEnv

env = UnityMultiAgentEnv(environment_filename, worker_id, default_visual)
```


* `environment_filename` refers to the path to the Unity environment.
* `worker_id` refers to the port to use for communication with the environment.
* `default_visual` refers to whether to use visual observations (True) or vector observations (False) as the default observation provided by the `reset` and `step` functions.

The returned environment `env` will function as a gym.

For more on using the gym interface, see our [Jupyter Notebook tutorial](python/notebooks/getting-started-gym.ipynb).

## Limitation

 * It is only possible to use an environment with a single Brain.
 * In mutli-agent environments launched with `UnityEnv`, only the first agent will be controllable.
 * By default the first visual observation is provided as the `observation`, if present. Otherwise vector observations are provided. 
 * All `BrainInfo` output from the environment can still be accessed from the `info` provided by `env.step(action)`.