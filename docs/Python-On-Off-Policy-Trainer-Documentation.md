# Table of Contents

* [mlagents.trainers.trainer.on\_policy\_trainer](#mlagents.trainers.trainer.on_policy_trainer)
  * [OnPolicyTrainer](#mlagents.trainers.trainer.on_policy_trainer.OnPolicyTrainer)
    * [\_\_init\_\_](#mlagents.trainers.trainer.on_policy_trainer.OnPolicyTrainer.__init__)
    * [add\_policy](#mlagents.trainers.trainer.on_policy_trainer.OnPolicyTrainer.add_policy)
* [mlagents.trainers.trainer.off\_policy\_trainer](#mlagents.trainers.trainer.off_policy_trainer)
  * [OffPolicyTrainer](#mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer)
    * [\_\_init\_\_](#mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.__init__)
    * [save\_model](#mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.save_model)
    * [save\_replay\_buffer](#mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.save_replay_buffer)
    * [load\_replay\_buffer](#mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.load_replay_buffer)
    * [add\_policy](#mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.add_policy)
* [mlagents.trainers.trainer.rl\_trainer](#mlagents.trainers.trainer.rl_trainer)
  * [RLTrainer](#mlagents.trainers.trainer.rl_trainer.RLTrainer)
    * [end\_episode](#mlagents.trainers.trainer.rl_trainer.RLTrainer.end_episode)
    * [create\_optimizer](#mlagents.trainers.trainer.rl_trainer.RLTrainer.create_optimizer)
    * [save\_model](#mlagents.trainers.trainer.rl_trainer.RLTrainer.save_model)
    * [advance](#mlagents.trainers.trainer.rl_trainer.RLTrainer.advance)
* [mlagents.trainers.trainer.trainer](#mlagents.trainers.trainer.trainer)
  * [Trainer](#mlagents.trainers.trainer.trainer.Trainer)
    * [\_\_init\_\_](#mlagents.trainers.trainer.trainer.Trainer.__init__)
    * [stats\_reporter](#mlagents.trainers.trainer.trainer.Trainer.stats_reporter)
    * [parameters](#mlagents.trainers.trainer.trainer.Trainer.parameters)
    * [get\_max\_steps](#mlagents.trainers.trainer.trainer.Trainer.get_max_steps)
    * [get\_step](#mlagents.trainers.trainer.trainer.Trainer.get_step)
    * [threaded](#mlagents.trainers.trainer.trainer.Trainer.threaded)
    * [should\_still\_train](#mlagents.trainers.trainer.trainer.Trainer.should_still_train)
    * [reward\_buffer](#mlagents.trainers.trainer.trainer.Trainer.reward_buffer)
    * [save\_model](#mlagents.trainers.trainer.trainer.Trainer.save_model)
    * [end\_episode](#mlagents.trainers.trainer.trainer.Trainer.end_episode)
    * [create\_policy](#mlagents.trainers.trainer.trainer.Trainer.create_policy)
    * [add\_policy](#mlagents.trainers.trainer.trainer.Trainer.add_policy)
    * [get\_policy](#mlagents.trainers.trainer.trainer.Trainer.get_policy)
    * [advance](#mlagents.trainers.trainer.trainer.Trainer.advance)
    * [publish\_policy\_queue](#mlagents.trainers.trainer.trainer.Trainer.publish_policy_queue)
    * [subscribe\_trajectory\_queue](#mlagents.trainers.trainer.trainer.Trainer.subscribe_trajectory_queue)
* [mlagents.trainers.settings](#mlagents.trainers.settings)
  * [deep\_update\_dict](#mlagents.trainers.settings.deep_update_dict)
  * [RewardSignalSettings](#mlagents.trainers.settings.RewardSignalSettings)
    * [structure](#mlagents.trainers.settings.RewardSignalSettings.structure)
  * [ParameterRandomizationSettings](#mlagents.trainers.settings.ParameterRandomizationSettings)
    * [\_\_str\_\_](#mlagents.trainers.settings.ParameterRandomizationSettings.__str__)
    * [structure](#mlagents.trainers.settings.ParameterRandomizationSettings.structure)
    * [unstructure](#mlagents.trainers.settings.ParameterRandomizationSettings.unstructure)
    * [apply](#mlagents.trainers.settings.ParameterRandomizationSettings.apply)
  * [ConstantSettings](#mlagents.trainers.settings.ConstantSettings)
    * [\_\_str\_\_](#mlagents.trainers.settings.ConstantSettings.__str__)
    * [apply](#mlagents.trainers.settings.ConstantSettings.apply)
  * [UniformSettings](#mlagents.trainers.settings.UniformSettings)
    * [\_\_str\_\_](#mlagents.trainers.settings.UniformSettings.__str__)
    * [apply](#mlagents.trainers.settings.UniformSettings.apply)
  * [GaussianSettings](#mlagents.trainers.settings.GaussianSettings)
    * [\_\_str\_\_](#mlagents.trainers.settings.GaussianSettings.__str__)
    * [apply](#mlagents.trainers.settings.GaussianSettings.apply)
  * [MultiRangeUniformSettings](#mlagents.trainers.settings.MultiRangeUniformSettings)
    * [\_\_str\_\_](#mlagents.trainers.settings.MultiRangeUniformSettings.__str__)
    * [apply](#mlagents.trainers.settings.MultiRangeUniformSettings.apply)
  * [CompletionCriteriaSettings](#mlagents.trainers.settings.CompletionCriteriaSettings)
    * [need\_increment](#mlagents.trainers.settings.CompletionCriteriaSettings.need_increment)
  * [Lesson](#mlagents.trainers.settings.Lesson)
  * [EnvironmentParameterSettings](#mlagents.trainers.settings.EnvironmentParameterSettings)
    * [structure](#mlagents.trainers.settings.EnvironmentParameterSettings.structure)
  * [TrainerSettings](#mlagents.trainers.settings.TrainerSettings)
    * [structure](#mlagents.trainers.settings.TrainerSettings.structure)
  * [CheckpointSettings](#mlagents.trainers.settings.CheckpointSettings)
    * [prioritize\_resume\_init](#mlagents.trainers.settings.CheckpointSettings.prioritize_resume_init)
  * [RunOptions](#mlagents.trainers.settings.RunOptions)
    * [from\_argparse](#mlagents.trainers.settings.RunOptions.from_argparse)

<a name="mlagents.trainers.trainer.on_policy_trainer"></a>
# mlagents.trainers.trainer.on\_policy\_trainer

<a name="mlagents.trainers.trainer.on_policy_trainer.OnPolicyTrainer"></a>
## OnPolicyTrainer Objects

```python
class OnPolicyTrainer(RLTrainer)
```

The PPOTrainer is an implementation of the PPO algorithm.

<a name="mlagents.trainers.trainer.on_policy_trainer.OnPolicyTrainer.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(behavior_name: str, reward_buff_cap: int, trainer_settings: TrainerSettings, training: bool, load: bool, seed: int, artifact_path: str)
```

Responsible for collecting experiences and training an on-policy model.

**Arguments**:

- `behavior_name`: The name of the behavior associated with trainer config
- `reward_buff_cap`: Max reward history to track in the reward buffer
- `trainer_settings`: The parameters for the trainer.
- `training`: Whether the trainer is set for training.
- `load`: Whether the model should be loaded.
- `seed`: The seed the model will be initialized with
- `artifact_path`: The directory within which to store artifacts from this trainer.

<a name="mlagents.trainers.trainer.on_policy_trainer.OnPolicyTrainer.add_policy"></a>
#### add\_policy

```python
 | add_policy(parsed_behavior_id: BehaviorIdentifiers, policy: Policy) -> None
```

Adds policy to trainer.

**Arguments**:

- `parsed_behavior_id`: Behavior identifiers that the policy should belong to.
- `policy`: Policy to associate with name_behavior_id.

<a name="mlagents.trainers.trainer.off_policy_trainer"></a>
# mlagents.trainers.trainer.off\_policy\_trainer

<a name="mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer"></a>
## OffPolicyTrainer Objects

```python
class OffPolicyTrainer(RLTrainer)
```

The SACTrainer is an implementation of the SAC algorithm, with support
for discrete actions and recurrent networks.

<a name="mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(behavior_name: str, reward_buff_cap: int, trainer_settings: TrainerSettings, training: bool, load: bool, seed: int, artifact_path: str)
```

Responsible for collecting experiences and training an off-policy model.

**Arguments**:

- `behavior_name`: The name of the behavior associated with trainer config
- `reward_buff_cap`: Max reward history to track in the reward buffer
- `trainer_settings`: The parameters for the trainer.
- `training`: Whether the trainer is set for training.
- `load`: Whether the model should be loaded.
- `seed`: The seed the model will be initialized with
- `artifact_path`: The directory within which to store artifacts from this trainer.

<a name="mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.save_model"></a>
#### save\_model

```python
 | save_model() -> None
```

Saves the final training model to memory
Overrides the default to save the replay buffer.

<a name="mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.save_replay_buffer"></a>
#### save\_replay\_buffer

```python
 | save_replay_buffer() -> None
```

Save the training buffer's update buffer to a pickle file.

<a name="mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.load_replay_buffer"></a>
#### load\_replay\_buffer

```python
 | load_replay_buffer() -> None
```

Loads the last saved replay buffer from a file.

<a name="mlagents.trainers.trainer.off_policy_trainer.OffPolicyTrainer.add_policy"></a>
#### add\_policy

```python
 | add_policy(parsed_behavior_id: BehaviorIdentifiers, policy: Policy) -> None
```

Adds policy to trainer.

<a name="mlagents.trainers.trainer.rl_trainer"></a>
# mlagents.trainers.trainer.rl\_trainer

<a name="mlagents.trainers.trainer.rl_trainer.RLTrainer"></a>
## RLTrainer Objects

```python
class RLTrainer(Trainer)
```

This class is the base class for trainers that use Reward Signals.

<a name="mlagents.trainers.trainer.rl_trainer.RLTrainer.end_episode"></a>
#### end\_episode

```python
 | end_episode() -> None
```

A signal that the Episode has ended. The buffer must be reset.
Get only called when the academy resets.

<a name="mlagents.trainers.trainer.rl_trainer.RLTrainer.create_optimizer"></a>
#### create\_optimizer

```python
 | @abc.abstractmethod
 | create_optimizer() -> TorchOptimizer
```

Creates an Optimizer object

<a name="mlagents.trainers.trainer.rl_trainer.RLTrainer.save_model"></a>
#### save\_model

```python
 | save_model() -> None
```

Saves the policy associated with this trainer.

<a name="mlagents.trainers.trainer.rl_trainer.RLTrainer.advance"></a>
#### advance

```python
 | advance() -> None
```

Steps the trainer, taking in trajectories and updates if ready.
Will block and wait briefly if there are no trajectories.

<a name="mlagents.trainers.trainer.trainer"></a>
# mlagents.trainers.trainer.trainer

<a name="mlagents.trainers.trainer.trainer.Trainer"></a>
## Trainer Objects

```python
class Trainer(abc.ABC)
```

This class is the base class for the mlagents_envs.trainers

<a name="mlagents.trainers.trainer.trainer.Trainer.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(brain_name: str, trainer_settings: TrainerSettings, training: bool, load: bool, artifact_path: str, reward_buff_cap: int = 1)
```

Responsible for collecting experiences and training a neural network model.

**Arguments**:

- `brain_name`: Brain name of brain to be trained.
- `trainer_settings`: The parameters for the trainer (dictionary).
- `training`: Whether the trainer is set for training.
- `artifact_path`: The directory within which to store artifacts from this trainer
- `reward_buff_cap`:

<a name="mlagents.trainers.trainer.trainer.Trainer.stats_reporter"></a>
#### stats\_reporter

```python
 | @property
 | stats_reporter()
```

Returns the stats reporter associated with this Trainer.

<a name="mlagents.trainers.trainer.trainer.Trainer.parameters"></a>
#### parameters

```python
 | @property
 | parameters() -> TrainerSettings
```

Returns the trainer parameters of the trainer.

<a name="mlagents.trainers.trainer.trainer.Trainer.get_max_steps"></a>
#### get\_max\_steps

```python
 | @property
 | get_max_steps() -> int
```

Returns the maximum number of steps. Is used to know when the trainer should be stopped.

**Returns**:

The maximum number of steps of the trainer

<a name="mlagents.trainers.trainer.trainer.Trainer.get_step"></a>
#### get\_step

```python
 | @property
 | get_step() -> int
```

Returns the number of steps the trainer has performed

**Returns**:

the step count of the trainer

<a name="mlagents.trainers.trainer.trainer.Trainer.threaded"></a>
#### threaded

```python
 | @property
 | threaded() -> bool
```

Whether or not to run the trainer in a thread. True allows the trainer to
update the policy while the environment is taking steps. Set to False to
enforce strict on-policy updates (i.e. don't update the policy when taking steps.)

<a name="mlagents.trainers.trainer.trainer.Trainer.should_still_train"></a>
#### should\_still\_train

```python
 | @property
 | should_still_train() -> bool
```

Returns whether or not the trainer should train. A Trainer could
stop training if it wasn't training to begin with, or if max_steps
is reached.

<a name="mlagents.trainers.trainer.trainer.Trainer.reward_buffer"></a>
#### reward\_buffer

```python
 | @property
 | reward_buffer() -> Deque[float]
```

Returns the reward buffer. The reward buffer contains the cumulative
rewards of the most recent episodes completed by agents using this
trainer.

**Returns**:

the reward buffer.

<a name="mlagents.trainers.trainer.trainer.Trainer.save_model"></a>
#### save\_model

```python
 | @abc.abstractmethod
 | save_model() -> None
```

Saves model file(s) for the policy or policies associated with this trainer.

<a name="mlagents.trainers.trainer.trainer.Trainer.end_episode"></a>
#### end\_episode

```python
 | @abc.abstractmethod
 | end_episode()
```

A signal that the Episode has ended. The buffer must be reset.
Get only called when the academy resets.

<a name="mlagents.trainers.trainer.trainer.Trainer.create_policy"></a>
#### create\_policy

```python
 | @abc.abstractmethod
 | create_policy(parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec) -> Policy
```

Creates a Policy object

<a name="mlagents.trainers.trainer.trainer.Trainer.add_policy"></a>
#### add\_policy

```python
 | @abc.abstractmethod
 | add_policy(parsed_behavior_id: BehaviorIdentifiers, policy: Policy) -> None
```

Adds policy to trainer.

<a name="mlagents.trainers.trainer.trainer.Trainer.get_policy"></a>
#### get\_policy

```python
 | get_policy(name_behavior_id: str) -> Policy
```

Gets policy associated with name_behavior_id

**Arguments**:

- `name_behavior_id`: Fully qualified behavior name

**Returns**:

Policy associated with name_behavior_id

<a name="mlagents.trainers.trainer.trainer.Trainer.advance"></a>
#### advance

```python
 | @abc.abstractmethod
 | advance() -> None
```

Advances the trainer. Typically, this means grabbing trajectories
from all subscribed trajectory queues (self.trajectory_queues), and updating
a policy using the steps in them, and if needed pushing a new policy onto the right
policy queues (self.policy_queues).

<a name="mlagents.trainers.trainer.trainer.Trainer.publish_policy_queue"></a>
#### publish\_policy\_queue

```python
 | publish_policy_queue(policy_queue: AgentManagerQueue[Policy]) -> None
```

Adds a policy queue to the list of queues to publish to when this Trainer
makes a policy update

**Arguments**:

- `policy_queue`: Policy queue to publish to.

<a name="mlagents.trainers.trainer.trainer.Trainer.subscribe_trajectory_queue"></a>
#### subscribe\_trajectory\_queue

```python
 | subscribe_trajectory_queue(trajectory_queue: AgentManagerQueue[Trajectory]) -> None
```

Adds a trajectory queue to the list of queues for the trainer to ingest Trajectories from.

**Arguments**:

- `trajectory_queue`: Trajectory queue to read from.

<a name="mlagents.trainers.settings"></a>
# mlagents.trainers.settings

<a name="mlagents.trainers.settings.deep_update_dict"></a>
#### deep\_update\_dict

```python
deep_update_dict(d: Dict, update_d: Mapping) -> None
```

Similar to dict.update(), but works for nested dicts of dicts as well.

<a name="mlagents.trainers.settings.RewardSignalSettings"></a>
## RewardSignalSettings Objects

```python
@attr.s(auto_attribs=True)
class RewardSignalSettings()
```

<a name="mlagents.trainers.settings.RewardSignalSettings.structure"></a>
#### structure

```python
 | @staticmethod
 | structure(d: Mapping, t: type) -> Any
```

Helper method to structure a Dict of RewardSignalSettings class. Meant to be registered with
cattr.register_structure_hook() and called with cattr.structure(). This is needed to handle
the special Enum selection of RewardSignalSettings classes.

<a name="mlagents.trainers.settings.ParameterRandomizationSettings"></a>
## ParameterRandomizationSettings Objects

```python
@attr.s(auto_attribs=True)
class ParameterRandomizationSettings(abc.ABC)
```

<a name="mlagents.trainers.settings.ParameterRandomizationSettings.__str__"></a>
#### \_\_str\_\_

```python
 | __str__() -> str
```

Helper method to output sampler stats to console.

<a name="mlagents.trainers.settings.ParameterRandomizationSettings.structure"></a>
#### structure

```python
 | @staticmethod
 | structure(d: Union[Mapping, float], t: type) -> "ParameterRandomizationSettings"
```

Helper method to a ParameterRandomizationSettings class. Meant to be registered with
cattr.register_structure_hook() and called with cattr.structure(). This is needed to handle
the special Enum selection of ParameterRandomizationSettings classes.

<a name="mlagents.trainers.settings.ParameterRandomizationSettings.unstructure"></a>
#### unstructure

```python
 | @staticmethod
 | unstructure(d: "ParameterRandomizationSettings") -> Mapping
```

Helper method to a ParameterRandomizationSettings class. Meant to be registered with
cattr.register_unstructure_hook() and called with cattr.unstructure().

<a name="mlagents.trainers.settings.ParameterRandomizationSettings.apply"></a>
#### apply

```python
 | @abc.abstractmethod
 | apply(key: str, env_channel: EnvironmentParametersChannel) -> None
```

Helper method to send sampler settings over EnvironmentParametersChannel
Calls the appropriate sampler type set method.

**Arguments**:

- `key`: environment parameter to be sampled
- `env_channel`: The EnvironmentParametersChannel to communicate sampler settings to environment

<a name="mlagents.trainers.settings.ConstantSettings"></a>
## ConstantSettings Objects

```python
@attr.s(auto_attribs=True)
class ConstantSettings(ParameterRandomizationSettings)
```

<a name="mlagents.trainers.settings.ConstantSettings.__str__"></a>
#### \_\_str\_\_

```python
 | __str__() -> str
```

Helper method to output sampler stats to console.

<a name="mlagents.trainers.settings.ConstantSettings.apply"></a>
#### apply

```python
 | apply(key: str, env_channel: EnvironmentParametersChannel) -> None
```

Helper method to send sampler settings over EnvironmentParametersChannel
Calls the constant sampler type set method.

**Arguments**:

- `key`: environment parameter to be sampled
- `env_channel`: The EnvironmentParametersChannel to communicate sampler settings to environment

<a name="mlagents.trainers.settings.UniformSettings"></a>
## UniformSettings Objects

```python
@attr.s(auto_attribs=True)
class UniformSettings(ParameterRandomizationSettings)
```

<a name="mlagents.trainers.settings.UniformSettings.__str__"></a>
#### \_\_str\_\_

```python
 | __str__() -> str
```

Helper method to output sampler stats to console.

<a name="mlagents.trainers.settings.UniformSettings.apply"></a>
#### apply

```python
 | apply(key: str, env_channel: EnvironmentParametersChannel) -> None
```

Helper method to send sampler settings over EnvironmentParametersChannel
Calls the uniform sampler type set method.

**Arguments**:

- `key`: environment parameter to be sampled
- `env_channel`: The EnvironmentParametersChannel to communicate sampler settings to environment

<a name="mlagents.trainers.settings.GaussianSettings"></a>
## GaussianSettings Objects

```python
@attr.s(auto_attribs=True)
class GaussianSettings(ParameterRandomizationSettings)
```

<a name="mlagents.trainers.settings.GaussianSettings.__str__"></a>
#### \_\_str\_\_

```python
 | __str__() -> str
```

Helper method to output sampler stats to console.

<a name="mlagents.trainers.settings.GaussianSettings.apply"></a>
#### apply

```python
 | apply(key: str, env_channel: EnvironmentParametersChannel) -> None
```

Helper method to send sampler settings over EnvironmentParametersChannel
Calls the gaussian sampler type set method.

**Arguments**:

- `key`: environment parameter to be sampled
- `env_channel`: The EnvironmentParametersChannel to communicate sampler settings to environment

<a name="mlagents.trainers.settings.MultiRangeUniformSettings"></a>
## MultiRangeUniformSettings Objects

```python
@attr.s(auto_attribs=True)
class MultiRangeUniformSettings(ParameterRandomizationSettings)
```

<a name="mlagents.trainers.settings.MultiRangeUniformSettings.__str__"></a>
#### \_\_str\_\_

```python
 | __str__() -> str
```

Helper method to output sampler stats to console.

<a name="mlagents.trainers.settings.MultiRangeUniformSettings.apply"></a>
#### apply

```python
 | apply(key: str, env_channel: EnvironmentParametersChannel) -> None
```

Helper method to send sampler settings over EnvironmentParametersChannel
Calls the multirangeuniform sampler type set method.

**Arguments**:

- `key`: environment parameter to be sampled
- `env_channel`: The EnvironmentParametersChannel to communicate sampler settings to environment

<a name="mlagents.trainers.settings.CompletionCriteriaSettings"></a>
## CompletionCriteriaSettings Objects

```python
@attr.s(auto_attribs=True)
class CompletionCriteriaSettings()
```

CompletionCriteriaSettings contains the information needed to figure out if the next
lesson must start.

<a name="mlagents.trainers.settings.CompletionCriteriaSettings.need_increment"></a>
#### need\_increment

```python
 | need_increment(progress: float, reward_buffer: List[float], smoothing: float) -> Tuple[bool, float]
```

Given measures, this method returns a boolean indicating if the lesson
needs to change now, and a float corresponding to the new smoothed value.

<a name="mlagents.trainers.settings.Lesson"></a>
## Lesson Objects

```python
@attr.s(auto_attribs=True)
class Lesson()
```

Gathers the data of one lesson for one environment parameter including its name,
the condition that must be fullfiled for the lesson to be completed and a sampler
for the environment parameter. If the completion_criteria is None, then this is
the last lesson in the curriculum.

<a name="mlagents.trainers.settings.EnvironmentParameterSettings"></a>
## EnvironmentParameterSettings Objects

```python
@attr.s(auto_attribs=True)
class EnvironmentParameterSettings()
```

EnvironmentParameterSettings is an ordered list of lessons for one environment
parameter.

<a name="mlagents.trainers.settings.EnvironmentParameterSettings.structure"></a>
#### structure

```python
 | @staticmethod
 | structure(d: Mapping, t: type) -> Dict[str, "EnvironmentParameterSettings"]
```

Helper method to structure a Dict of EnvironmentParameterSettings class. Meant
to be registered with cattr.register_structure_hook() and called with
cattr.structure().

<a name="mlagents.trainers.settings.TrainerSettings"></a>
## TrainerSettings Objects

```python
@attr.s(auto_attribs=True)
class TrainerSettings(ExportableSettings)
```

<a name="mlagents.trainers.settings.TrainerSettings.structure"></a>
#### structure

```python
 | @staticmethod
 | structure(d: Mapping, t: type) -> Any
```

Helper method to structure a TrainerSettings class. Meant to be registered with
cattr.register_structure_hook() and called with cattr.structure().

<a name="mlagents.trainers.settings.CheckpointSettings"></a>
## CheckpointSettings Objects

```python
@attr.s(auto_attribs=True)
class CheckpointSettings()
```

<a name="mlagents.trainers.settings.CheckpointSettings.prioritize_resume_init"></a>
#### prioritize\_resume\_init

```python
 | prioritize_resume_init() -> None
```

Prioritize explicit command line resume/init over conflicting yaml options.
if both resume/init are set at one place use resume

<a name="mlagents.trainers.settings.RunOptions"></a>
## RunOptions Objects

```python
@attr.s(auto_attribs=True)
class RunOptions(ExportableSettings)
```

<a name="mlagents.trainers.settings.RunOptions.from_argparse"></a>
#### from\_argparse

```python
 | @staticmethod
 | from_argparse(args: argparse.Namespace) -> "RunOptions"
```

Takes an argparse.Namespace as specified in `parse_command_line`, loads input configuration files
from file paths, and converts to a RunOptions instance.

**Arguments**:

- `args`: collection of command-line parameters passed to mlagents-learn

**Returns**:

RunOptions representing the passed in arguments, with trainer config, curriculum and sampler
configs loaded from files.
