### Step 1: Write your own custom trainer class
Before you start writing your code, make sure to create a python environment:
```shell
conda create -n trainer-env python=3.8
```

Users of the plug-in system are responsible for implementing the trainer class subject to the API standard. Let us follow an example by implementing a custom trainer named "YourCustomTrainer". You can either extend `OnPolicyTrainer` or `OffPolicyTrainer` classes depending on the training strategies you choose.

Model-free RL algorithms generally fall into two broad categories: on-policy and off-policy. On-policy algorithms rely on performing updates based on data gathered from the current policy. Off-policy algorithms learn a Q function from a buffer of previous data, then use this Q function to make decisions. Off-policy algorithms have three key benefits in the context of ML-Agents:
They tend to use fewer samples than on-policy as they can pull and re-use data from the buffer many times.
They allow player demonstrations to be inserted in-line with RL data into the buffer, enabling new ways of doing imitation learning by streaming player data.
They are conducive to distributed training, where the policy running on other machines may not be synchronized with the current policy.
However, until recently, off-policy algorithms tended to be more brittle, had difficulty with exploration, and were usually not as useful for continuous control problems. Soft Actor-Critic (Haarnoja et. al, 2018) is an off-policy algorithm that combines the sample-efficiency of Q-learning with the stochasticity of a policy-gradient method such as PPO.

Your custom trainers are Responsible for collecting experiences and training the models. Your custom trainer class acts like a co-ordinator to the policy and optimizer. To start implement methods in the class, create a policy and an optimizer class objects:


```python
def create_policy(
    self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
) -> TorchPolicy:

    actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
    actor_kwargs: Dict[str, Any] = {
        "conditional_sigma": False,
        "tanh_squash": False,
    }
    if self.shared_critic:
        reward_signal_configs = self.trainer_settings.reward_signals
        reward_signal_names = [
            key.value for key, _ in reward_signal_configs.items()
        ]
        actor_cls = SharedActorCritic
        actor_kwargs.update({"stream_names": reward_signal_names})

    policy = TorchPolicy(
        self.seed,
        behavior_spec,
        self.trainer_settings.network_settings,
        actor_cls,
        actor_kwargs,
    )
    return policy

```

Depending on whether you use shared or separate network architecuture for your policy, we provide `SimpleActor` and `SharedActorCritic` from `mlagents.trainers.torch_entities.networks` that you can choose from. In our example above, we use a `SimpleActor`

Next, create an optimizer class object from `create_optimizer` method:


```python
def create_optimizer(self) -> TorchOptimizer:
    return TorchPPOOptimizer(  # type: ignore
        cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
    )  # type: ignore

```

There are a couple abstract methods(`_process_trajectory` and `_update_policy`) inherited from `RLTrainer` you need to implement in your custom trainer class. `_process_trajectory` takes a trajectory and processes it, puts it into the update buffer. Processing involves calculating value and advantage targets for the model updating step. Given input `trajectory: Trajectory`, users are responsible for processing the data in the trajectory and append `agent_buffer_trajectory` to the back of update buffer by calling `self._append_to_update_buffer(agent_buffer_trajectory)`, whose output will be used in updating the model in `optimizer` class.

A typical `_process_trajectory` function(incomplete) - would look like the following:
```python
def _process_trajectory(self, trajectory: Trajectory) -> None:
    super()._process_trajectory(trajectory)
    agent_id = trajectory.agent_id  # All the agents should have the same ID

    agent_buffer_trajectory = trajectory.to_agentbuffer()

    # Get all value estimates
    (
        value_estimates,
        value_next,
        value_memories,
    ) = self.optimizer.get_trajectory_value_estimates(
        agent_buffer_trajectory,
        trajectory.next_obs,
        trajectory.done_reached and not trajectory.interrupted,
    )

    for name, v in value_estimates.items():
        agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
            v
        )
        self._stats_reporter.add_stat(
            f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
            np.mean(v),
        )

    # Evaluate all reward functions
    self.collected_rewards["environment"][agent_id] += np.sum(
        agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
    )
    for name, reward_signal in self.optimizer.reward_signals.items():
        evaluate_result = (
            reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
        )
        agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
            evaluate_result
        )
        # Report the reward signals
        self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

    self._append_to_update_buffer(agent_buffer_trajectory)

```

A trajectory will be a list of dictionaries of string to anything. When calling forward on a policy, the argument will include an “experience” dict of string to anything from the last step. The forward method will generate action and the next “experience” dictionary. Examples of fields in the “experience” dictionary include observation, action, reward, done status, group_reward, LSTM memory state, etc...



### Step 2: implement your custom optimizer for the trainer.
We will show you an example we implemented - `class TorchPPOOptimizer(TorchOptimizer)`, Which Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy. Your optimizer should include a value estimator and a loss function in the update method

Before writing your optimizer class, first define setting class `class PPOSettings(OnPolicyHyperparamSettings):
` for your custom optimizer:



```python
class PPOSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR

```

You should implement `update` function:


```python
def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:

```

Calculate losses and other metrics from an `AgentBuffer` generated from your trainer class, a typical pattern(incomplete) would like this:


```python
run_out = self.policy.actor.get_stats(
    current_obs,
    actions,
    masks=act_masks,
    memories=memories,
    sequence_length=self.policy.sequence_length,
)

log_probs = run_out["log_probs"]
entropy = run_out["entropy"]

values, _ = self.critic.critic_pass(
    current_obs,
    memories=value_memories,
    sequence_length=self.policy.sequence_length,
)
policy_loss = ModelUtils.trust_region_policy_loss(
    ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
    log_probs,
    old_log_probs,
    loss_masks,
    decay_eps,
)
loss = (
    policy_loss
    + 0.5 * value_loss
    - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
)

```

Update the model and return  the a dictionary including calculated losses and updated decay learning rate:


```python
ModelUtils.update_learning_rate(self.optimizer, decay_lr)
self.optimizer.zero_grad()
loss.backward()

self.optimizer.step()
update_stats = {
    # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
    # TODO: After PyTorch is default, change to something more correct.
    "Losses/Policy Loss": torch.abs(policy_loss).item(),
    "Losses/Value Loss": value_loss.item(),
    "Policy/Learning Rate": decay_lr,
    "Policy/Epsilon": decay_eps,
    "Policy/Beta": decay_bet,
}

```

### Step 3: Integrate your custom trainer into the plugin system

By integrating a custom trainer into the plugin system, a user can use their published packages which have their implementations. To do that, you need to add a setup.py file. In the call to setup(), you'll need to add to the entry_points dictionary for each plugin interface that you implement. The form of this is {entry point name}={plugin module}:{plugin function}. For example:



```python
entry_points={
        ML_AGENTS_TRAINER_TYPE: [
            "your_trainer_type=your_package.your_custom_trainer:get_type_and_setting"
        ]
    },
```

Some key elements in the code:

```
ML_AGENTS_TRAINER_TYPE: a string constant for trainer type
your_trainer_type: name your trainer type, used in configuration file
your_package: your pip installable package containing custom trainer implementation
```

Also define get_type_and_setting method in YourCustomTrainer class:


```python
def get_type_and_setting():
    return {YourCustomTrainer.get_trainer_name(): YourCustomTrainer}, {
        YourCustomTrainer.get_trainer_name(): YourCustomSetting
    }

```

Finally, specify trainer type in the config file:


```python
behaviors:
  3DBall:
    trainer_type: your_trainer_type
...
```

### Step 4: Install your custom trainer and run training:
Before installing your custom trainer package, make sure you have `ml-agents-env` and `ml-agents` installed

```shell
pip3 install -e ./ml-agents-envs && pip3 install -e ./ml-agents
```

Install your cutom trainer package(if your package is pip installable):
```shell
pip3 install your_custom_package
```
Or follow our internal implementations:
```shell
pip install -e ./ml-agents-trainer-plugin
```

Following the previous installations your package is added as an entrypoint and you can use a config file with new
trainers:
```shell
mlagents-learn ml-agents-trainer-plugin/mlagents_trainer_plugin/a2c/a2c_3DBall.yaml --run-id <run-id-name>
--env <env-executable>
```

### Validate your implementations:
Create a clean python environment with python 3.8+ before you start.
```shell
conda create -n trainer-env python=3.8
```

Make sure you follow previous steps and install all required packages. We are testing internal implementations here, but ML-Agents users can run similar validations once they have their own implementation installed:
```shell
pip3 install -e ./ml-agents-envs && pip3 install -e ./ml-agents
pip install -e ./ml-agents-trainer-plugin
```
Once your package is added as an entrypoint and you can use a config file with new trainer. Check if trainer type is specified in the config file `a2c_3DBall.yaml`:
```
trainer_type: a2c
```

Test if custom trainer package is install:
```shell
mlagents-learn ml-agents-trainer-plugin/mlagents_trainer_plugin/a2c/a2c_3DBall.yaml --run-id test-trainer
```

If it is properly installed, you will see Unity logo and message indicating training will start:
```
[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
```

If you see the following error message, it could be due to train type is wrong or the trainer type specified is not installed:
```shell
mlagents.trainers.exception.TrainerConfigError: Invalid trainer type a2c was found
```

You can also check all trainers installed in the registry. Type `python` in your shell to open a REPL session. Run the python code below, you should be able to see all trainer types installed:
```python
>>> import pkg_resources
>>> for entry in pkg_resources.iter_entry_points('mlagents.trainer_type'):
...     print(entry)
...
default = mlagents.plugins.trainer_type:get_default_trainer_types
a2c = mlagents_trainer_plugin.a2c.a2c_trainer:get_type_and_setting
dqn = mlagents_trainer_plugin.dqn.dqn_trainer:get_type_and_setting
```