# Table of Contents

* [mlagents.trainers.optimizer.torch\_optimizer](#mlagents.trainers.optimizer.torch_optimizer)
  * [TorchOptimizer](#mlagents.trainers.optimizer.torch_optimizer.TorchOptimizer)
    * [create\_reward\_signals](#mlagents.trainers.optimizer.torch_optimizer.TorchOptimizer.create_reward_signals)
    * [get\_trajectory\_value\_estimates](#mlagents.trainers.optimizer.torch_optimizer.TorchOptimizer.get_trajectory_value_estimates)
* [mlagents.trainers.optimizer.optimizer](#mlagents.trainers.optimizer.optimizer)
  * [Optimizer](#mlagents.trainers.optimizer.optimizer.Optimizer)
    * [update](#mlagents.trainers.optimizer.optimizer.Optimizer.update)

<a name="mlagents.trainers.optimizer.torch_optimizer"></a>
# mlagents.trainers.optimizer.torch\_optimizer

<a name="mlagents.trainers.optimizer.torch_optimizer.TorchOptimizer"></a>
## TorchOptimizer Objects

```python
class TorchOptimizer(Optimizer)
```

<a name="mlagents.trainers.optimizer.torch_optimizer.TorchOptimizer.create_reward_signals"></a>
#### create\_reward\_signals

```python
 | create_reward_signals(reward_signal_configs: Dict[RewardSignalType, RewardSignalSettings]) -> None
```

Create reward signals

**Arguments**:

- `reward_signal_configs`: Reward signal config.

<a name="mlagents.trainers.optimizer.torch_optimizer.TorchOptimizer.get_trajectory_value_estimates"></a>
#### get\_trajectory\_value\_estimates

```python
 | get_trajectory_value_estimates(batch: AgentBuffer, next_obs: List[np.ndarray], done: bool, agent_id: str = "") -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]
```

Get value estimates and memories for a trajectory, in batch form.

**Arguments**:

- `batch`: An AgentBuffer that consists of a trajectory.
- `next_obs`: the next observation (after the trajectory). Used for boostrapping
    if this is not a termiinal trajectory.
- `done`: Set true if this is a terminal trajectory.
- `agent_id`: Agent ID of the agent that this trajectory belongs to.

**Returns**:

A Tuple of the Value Estimates as a Dict of [name, np.ndarray(trajectory_len)],
    the final value estimate as a Dict of [name, float], and optionally (if using memories)
    an AgentBufferField of initial critic memories to be used during update.

<a name="mlagents.trainers.optimizer.optimizer"></a>
# mlagents.trainers.optimizer.optimizer

<a name="mlagents.trainers.optimizer.optimizer.Optimizer"></a>
## Optimizer Objects

```python
class Optimizer(abc.ABC)
```

Creates loss functions and auxillary networks (e.g. Q or Value) needed for training.
Provides methods to update the Policy.

<a name="mlagents.trainers.optimizer.optimizer.Optimizer.update"></a>
#### update

```python
 | @abc.abstractmethod
 | update(batch: AgentBuffer, num_sequences: int) -> Dict[str, float]
```

Update the Policy based on the batch that was passed in.

**Arguments**:

- `batch`: AgentBuffer that contains the minibatch of data used for this update.
- `num_sequences`: Number of recurrent sequences found in the minibatch.

**Returns**:

A Dict containing statistics (name, value) from the update (e.g. loss)
