# Unity ML-Agents Gym Wrapper

A common way in which machine learning researchers interact with simulation environments is via a wrapper provided by OpenAI called `gym`. For more information on the gym interface, see [here](https://github.com/openai/gym). 

We provide a gym wrapper, and instructions for using it with existing research projects which utilize gyms. 

## Using the Gym Wrapper
The gym wrapper is available from `python/environment/unity_gym_env.py`. To launch an environmnent from the root of the project repository use:

```python
from unityagents.unity_gym_env import UnityGymEnv

env = UnityGymEnv(environment_filename, worker_id, default_visual)
```

* `environment_filename` refers to the path to the Unity environment.
* `worker_id` refers to the port to use for communication with the environment.
* `default_visual` refers to whether to use visual observations (True) or vector observations (False) as the default observation provided by the `reset` and `step` functions.

The returned environment `env` will function as a gym.

For more on using the gym wrapper, see our [Jupyter Notebook tutorial](./python/Basics-Gym.ipynb).

## Limitation

 * It is only possible to use an environment with a single Brain.
 * Only first agent in first external Brain will be exposed via API.
 * By default the first visual observation is provided as the `observation`, if present. Otherwise vector observations are provided.  
 * All other output from the environment can still be accessed from the `info` provided by `env.reset()` and `env.step(action)`.