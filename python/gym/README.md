# ML-Agents Gym Wrapper

## `gym_wrapper.py`
First draft on a gym wrapper for ML-Agents. To launch an environmnent do :

```python
raw_env = UnityEnvironment(<env-name>)
env = GymWrapper(raw_env)
```

The environment `env` will behave like a gym.

__Limitations :__

 * Only works with environments containing one external brain
 * Only works with environments containing one agent
