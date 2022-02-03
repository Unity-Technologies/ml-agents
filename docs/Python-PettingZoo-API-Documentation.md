# Table of Contents

* [mlagents\_envs.envs.pettingzoo\_env\_factory](#mlagents_envs.envs.pettingzoo_env_factory)
  * [PettingZooEnvFactory](#mlagents_envs.envs.pettingzoo_env_factory.PettingZooEnvFactory)
    * [env](#mlagents_envs.envs.pettingzoo_env_factory.PettingZooEnvFactory.env)
* [mlagents\_envs.envs.unity\_aec\_env](#mlagents_envs.envs.unity_aec_env)
  * [UnityAECEnv](#mlagents_envs.envs.unity_aec_env.UnityAECEnv)
    * [\_\_init\_\_](#mlagents_envs.envs.unity_aec_env.UnityAECEnv.__init__)
    * [step](#mlagents_envs.envs.unity_aec_env.UnityAECEnv.step)
    * [observe](#mlagents_envs.envs.unity_aec_env.UnityAECEnv.observe)
    * [last](#mlagents_envs.envs.unity_aec_env.UnityAECEnv.last)
* [mlagents\_envs.envs.unity\_parallel\_env](#mlagents_envs.envs.unity_parallel_env)
  * [UnityParallelEnv](#mlagents_envs.envs.unity_parallel_env.UnityParallelEnv)
    * [\_\_init\_\_](#mlagents_envs.envs.unity_parallel_env.UnityParallelEnv.__init__)
    * [reset](#mlagents_envs.envs.unity_parallel_env.UnityParallelEnv.reset)
* [mlagents\_envs.envs.unity\_pettingzoo\_base\_env](#mlagents_envs.envs.unity_pettingzoo_base_env)
  * [UnityPettingzooBaseEnv](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv)
    * [observation\_spaces](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.observation_spaces)
    * [observation\_space](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.observation_space)
    * [action\_spaces](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.action_spaces)
    * [action\_space](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.action_space)
    * [side\_channel](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.side_channel)
    * [reset](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.reset)
    * [seed](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.seed)
    * [render](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.render)
    * [close](#mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.close)

<a name="mlagents_envs.envs.pettingzoo_env_factory"></a>
# mlagents\_envs.envs.pettingzoo\_env\_factory

<a name="mlagents_envs.envs.pettingzoo_env_factory.PettingZooEnvFactory"></a>
## PettingZooEnvFactory Objects

```python
class PettingZooEnvFactory()
```

<a name="mlagents_envs.envs.pettingzoo_env_factory.PettingZooEnvFactory.env"></a>
#### env

```python
 | env(seed: Optional[int] = None, **kwargs: Union[List, int, bool, None]) -> UnityAECEnv
```

Creates the environment with env_id from unity's default_registry and wraps it in a UnityToPettingZooWrapper

**Arguments**:

- `seed`: The seed for the action spaces of the agents.
- `kwargs`: Any argument accepted by `UnityEnvironment`class except file_name

<a name="mlagents_envs.envs.unity_aec_env"></a>
# mlagents\_envs.envs.unity\_aec\_env

<a name="mlagents_envs.envs.unity_aec_env.UnityAECEnv"></a>
## UnityAECEnv Objects

```python
class UnityAECEnv(UnityPettingzooBaseEnv,  AECEnv)
```

Unity AEC (PettingZoo) environment wrapper.

<a name="mlagents_envs.envs.unity_aec_env.UnityAECEnv.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(env: BaseEnv, seed: Optional[int] = None)
```

Initializes a Unity AEC environment wrapper.

**Arguments**:

- `env`: The UnityEnvironment that is being wrapped.
- `seed`: The seed for the action spaces of the agents.

<a name="mlagents_envs.envs.unity_aec_env.UnityAECEnv.step"></a>
#### step

```python
 | step(action: Any) -> None
```

Sets the action of the active agent and get the observation, reward, done
and info of the next agent.

**Arguments**:

- `action`: The action for the active agent

<a name="mlagents_envs.envs.unity_aec_env.UnityAECEnv.observe"></a>
#### observe

```python
 | observe(agent_id)
```

Returns the observation an agent currently can make. `last()` calls this function.

<a name="mlagents_envs.envs.unity_aec_env.UnityAECEnv.last"></a>
#### last

```python
 | last(observe=True)
```

returns observation, cumulative reward, done, info for the current agent (specified by self.agent_selection)

<a name="mlagents_envs.envs.unity_parallel_env"></a>
# mlagents\_envs.envs.unity\_parallel\_env

<a name="mlagents_envs.envs.unity_parallel_env.UnityParallelEnv"></a>
## UnityParallelEnv Objects

```python
class UnityParallelEnv(UnityPettingzooBaseEnv,  ParallelEnv)
```

Unity Parallel (PettingZoo) environment wrapper.

<a name="mlagents_envs.envs.unity_parallel_env.UnityParallelEnv.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(env: BaseEnv, seed: Optional[int] = None)
```

Initializes a Unity Parallel environment wrapper.

**Arguments**:

- `env`: The UnityEnvironment that is being wrapped.
- `seed`: The seed for the action spaces of the agents.

<a name="mlagents_envs.envs.unity_parallel_env.UnityParallelEnv.reset"></a>
#### reset

```python
 | reset() -> Dict[str, Any]
```

Resets the environment.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env"></a>
# mlagents\_envs.envs.unity\_pettingzoo\_base\_env

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv"></a>
## UnityPettingzooBaseEnv Objects

```python
class UnityPettingzooBaseEnv()
```

Unity Petting Zoo base environment.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.observation_spaces"></a>
#### observation\_spaces

```python
 | @property
 | observation_spaces() -> Dict[str, spaces.Space]
```

Return the observation spaces of all the agents.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.observation_space"></a>
#### observation\_space

```python
 | observation_space(agent: str) -> Optional[spaces.Space]
```

The observation space of the current agent.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.action_spaces"></a>
#### action\_spaces

```python
 | @property
 | action_spaces() -> Dict[str, spaces.Space]
```

Return the action spaces of all the agents.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.action_space"></a>
#### action\_space

```python
 | action_space(agent: str) -> Optional[spaces.Space]
```

The action space of the current agent.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.side_channel"></a>
#### side\_channel

```python
 | @property
 | side_channel() -> Dict[str, Any]
```

The side channels of the environment. You can access the side channels
of an environment with `env.side_channel[<name-of-channel>]`.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.reset"></a>
#### reset

```python
 | reset()
```

Resets the environment.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.seed"></a>
#### seed

```python
 | seed(seed=None)
```

Reseeds the environment (making the resulting environment deterministic).
`reset()` must be called after `seed()`, and before `step()`.

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.render"></a>
#### render

```python
 | render(mode="human")
```

NOT SUPPORTED.

Displays a rendered frame from the environment, if supported.
Alternate render modes in the default environments are `'rgb_array'`
which returns a numpy array and is supported by all environments outside of classic,
and `'ansi'` which returns the strings printed (specific to classic environments).

<a name="mlagents_envs.envs.unity_pettingzoo_base_env.UnityPettingzooBaseEnv.close"></a>
#### close

```python
 | close() -> None
```

Close the environment.
