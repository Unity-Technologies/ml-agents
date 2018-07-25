# Unity ML-Agents Gym Wrapper

A common way in which researchers interact with simulation environments is via wrapper provided by OpenAI called `gym`. Here we provide a gym wrapper, and instructions for using it with existing research projects which utilize gyms. 

## `unity_gym.py`
First draft on a gym wrapper for ML-Agents. To launch an environmnent use :

```python
env = GymWrapper(environment_filename, worker_id, default_visual)
```

* `environment_filename` refers to the path to the Unity environment.
* `worker_id` refers to the port to use for communication with the environment.
* `default_visual` refers to whether to use visual observations (True) or vector observations (False) as the default observation provided by the `reset` and `step` functions.

The returned environment `env` will function as a gym.

__Limitations :__

 * Only first agent in first external brain will be exposed via API.
 * By default the first visual observation is provided as the `observation`, if present. Otherwise vector observations are provided.  
 * All other output from the environment can still be accessed from the `info` provided by `env.reset()` and `env.step(action)`.