# Table of Contents

* [mlagents\_envs.envs.unity\_gym\_env](#mlagents_envs.envs.unity_gym_env)
  * [UnityGymException](#mlagents_envs.envs.unity_gym_env.UnityGymException)
  * [UnityToGymWrapper](#mlagents_envs.envs.unity_gym_env.UnityToGymWrapper)
    * [\_\_init\_\_](#mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.__init__)
    * [reset](#mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.reset)
    * [step](#mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.step)
    * [render](#mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.render)
    * [close](#mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.close)
    * [seed](#mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.seed)
  * [ActionFlattener](#mlagents_envs.envs.unity_gym_env.ActionFlattener)
    * [\_\_init\_\_](#mlagents_envs.envs.unity_gym_env.ActionFlattener.__init__)
    * [lookup\_action](#mlagents_envs.envs.unity_gym_env.ActionFlattener.lookup_action)

<a name="mlagents_envs.envs.unity_gym_env"></a>
# mlagents\_envs.envs.unity\_gym\_env

<a name="mlagents_envs.envs.unity_gym_env.UnityGymException"></a>
## UnityGymException Objects

```python
class UnityGymException(error.Error)
```

Any error related to the gym wrapper of ml-agents.

<a name="mlagents_envs.envs.unity_gym_env.UnityToGymWrapper"></a>
## UnityToGymWrapper Objects

```python
class UnityToGymWrapper(gym.Env)
```

Provides Gym wrapper for Unity Learning Environments.

<a name="mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(unity_env: BaseEnv, uint8_visual: bool = False, flatten_branched: bool = False, allow_multiple_obs: bool = False, action_space_seed: Optional[int] = None)
```

Environment initialization

**Arguments**:

- `unity_env`: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
- `uint8_visual`: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
- `flatten_branched`: If True, turn branched discrete action spaces into a Discrete space rather than
    MultiDiscrete.
- `allow_multiple_obs`: If True, return a list of np.ndarrays as observations with the first elements
    containing the visual observations and the last element containing the array of vector observations.
    If False, returns a single np.ndarray containing either only a single visual observation or the array of
    vector observations.
- `action_space_seed`: If non-None, will be used to set the random seed on created gym.Space instances.

<a name="mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.reset"></a>
#### reset

```python
 | reset() -> Union[List[np.ndarray], np.ndarray]
```

Resets the state of the environment and returns an initial observation.
Returns: observation (object/list): the initial observation of the
space.

<a name="mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.step"></a>
#### step

```python
 | step(action: List[Any]) -> GymStepResult
```

Run one timestep of the environment's dynamics. When end of
episode is reached, you are responsible for calling `reset()`
to reset this environment's state.
Accepts an action and returns a tuple (observation, reward, done, info).

**Arguments**:

- `action` _object/list_ - an action provided by the environment

**Returns**:

- `observation` _object/list_ - agent's observation of the current environment
  reward (float/list) : amount of reward returned after previous action
- `done` _boolean/list_ - whether the episode has ended.
- `info` _dict_ - contains auxiliary diagnostic information.

<a name="mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.render"></a>
#### render

```python
 | render(mode="rgb_array")
```

Return the latest visual observations.
Note that it will not render a new frame of the environment.

<a name="mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.close"></a>
#### close

```python
 | close() -> None
```

Override _close in your subclass to perform any necessary cleanup.
Environments will automatically close() themselves when
garbage collected or when the program exits.

<a name="mlagents_envs.envs.unity_gym_env.UnityToGymWrapper.seed"></a>
#### seed

```python
 | seed(seed: Any = None) -> None
```

Sets the seed for this env's random number generator(s).
Currently not implemented.

<a name="mlagents_envs.envs.unity_gym_env.ActionFlattener"></a>
## ActionFlattener Objects

```python
class ActionFlattener()
```

Flattens branched discrete action spaces into single-branch discrete action spaces.

<a name="mlagents_envs.envs.unity_gym_env.ActionFlattener.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(branched_action_space)
```

Initialize the flattener.

**Arguments**:

- `branched_action_space`: A List containing the sizes of each branch of the action
space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.

<a name="mlagents_envs.envs.unity_gym_env.ActionFlattener.lookup_action"></a>
#### lookup\_action

```python
 | lookup_action(action)
```

Convert a scalar discrete action into a unique set of branched actions.

**Arguments**:

- `action`: A scalar value representing one of the discrete actions.

**Returns**:

The List containing the branched actions.
