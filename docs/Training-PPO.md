# Training with Proximal Policy Optimization

To train an agent, you will need to provide the agent one or more reward signals which
the agent should attempt to maximize. See [Reward Signals](Reward-Signals.md)
for the available reward signals and the corresponding hyperparameters.

If you are using the recurrent neural network (RNN) to utilize memory, see
[Using Recurrent Neural Networks](Feature-Memory.md) for RNN-specific training
details.

If you are using curriculum training to pace the difficulty of the learning task
presented to an agent, see [Training with Curriculum
Learning](Training-Curriculum-Learning.md).

For information about imitation learning from demonstrations, see
[Training with Imitation Learning](Training-Imitation-Learning.md).

## Best Practices Training with PPO

## Hyperparameters

### Reward Signals

In reinforcement learning, the goal is to learn a Policy that maximizes reward.
At a base level, the reward is given by the environment. However, we could imagine
rewarding the agent for various different behaviors. For instance, we could reward
the agent for exploring new states, rather than just when an explicit reward is given.
Furthermore, we could mix reward signals to help the learning process.

Using `reward_signals` allows you to define [reward signals.](Reward-Signals.md)
The ML-Agents Toolkit provides three reward signals by default, the Extrinsic (environment)
reward signal, the Curiosity reward signal, which can be used to encourage exploration in
sparse extrinsic reward environments, and the GAIL reward signal. Please see [Reward Signals](Reward-Signals.md)
for additional details.

## (Optional) Recurrent Neural Network Hyperparameters

The below hyperparameters are only used when `use_recurrent` is set to true.

### Sequence Length

`sequence_length` corresponds to the length of the sequences of experience
passed through the network during training. This should be long enough to
capture whatever information your agent might need to remember over time. For
example, if your agent needs to remember the velocity of objects, then this can
be a small value. If your agent needs to remember a piece of information given
only once at the beginning of an episode, then this should be a larger value.

Typical Range: `4` - `128`

### Memory Size

`memory_size` corresponds to the size of the array of floating point numbers
used to store the hidden state of the recurrent neural network of the policy. This value must
be a multiple of 2, and should scale with the amount of information you expect
the agent will need to remember in order to successfully complete the task.

Typical Range: `32` - `256`

## (Optional) Behavioral Cloning Using Demonstrations

In some cases, you might want to bootstrap the agent's policy using behavior recorded
from a player. This can help guide the agent towards the reward. Behavioral Cloning (BC) adds
training operations that mimic a demonstration rather than attempting to maximize reward.

To use BC, add a `behavioral_cloning` section to the trainer_config. For instance:

```
    behavioral_cloning:
        demo_path: ./Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
        strength: 0.5
        steps: 10000
```

Below are the available hyperparameters for BC.

### Strength

`strength` corresponds to the learning rate of the imitation relative to the learning
rate of PPO, and roughly corresponds to how strongly we allow BC
to influence the policy.

Typical Range: `0.1` - `0.5`

### Demo Path

`demo_path` is the path to your `.demo` file or directory of `.demo` files.
See the [imitation learning guide](Training-Imitation-Learning.md) for more on `.demo` files.

### Steps

During BC, it is often desirable to stop using demonstrations after the agent has
"seen" rewards, and allow it to optimize past the available demonstrations and/or generalize
outside of the provided demonstrations. `steps` corresponds to the training steps over which
BC is active. The learning rate of BC will anneal over the steps. Set
the steps to 0 for constant imitation over the entire training run.

### (Optional) Batch Size

`batch_size` is the number of demonstration experiences used for one iteration of a gradient
descent update. If not specified, it will default to the `batch_size` defined for PPO.

Typical Range (Continuous): `512` - `5120`

Typical Range (Discrete): `32` - `512`

### (Optional) Number of Epochs

`num_epoch` is the number of passes through the experience buffer during
gradient descent. If not specified, it will default to the number of epochs set for PPO.

Typical Range: `3` - `10`

### (Optional) Samples Per Update

`samples_per_update` is the maximum number of samples
to use during each imitation update. You may want to lower this if your demonstration
dataset is very large to avoid overfitting the policy on demonstrations. Set to 0
to train over all of the demonstrations at each update step.

Default Value: `0` (all)

Typical Range: Approximately equal to PPO's `buffer_size`

### (Optional) Advanced: Disable Threading

By default, PPO model updates can happen while the environment is being stepped. This violates the
[on-policy](https://spinningup.openai.com/en/latest/user/algorithms.html#the-on-policy-algorithms)
assumption of PPO slightly in exchange for a 10-20% training speedup. To maintain the
strict on-policyness of PPO, you can disable parallel updates by setting `threaded` to `false`.

Default Value: `true`

## Training Statistics

To view training statistics, use TensorBoard. For information on launching and
using TensorBoard, see
[here](./Getting-Started.md#observing-training-progress).

### Cumulative Reward

The general trend in reward should consistently increase over time. Small ups
and downs are to be expected. Depending on the complexity of the task, a
significant increase in reward may not present itself until millions of steps
into the training process.

### Entropy

This corresponds to how random the decisions are. This should
consistently decrease during training. If it decreases too soon or not at all,
`beta` should be adjusted (when using discrete action space).

### Learning Rate

This will decrease over time on a linear schedule by default, unless `learning_rate_schedule`
is set to `constant`.

### Policy Loss

These values will oscillate during training. Generally they should be less than
1.0.

### Value Estimate

These values should increase as the cumulative reward increases. They correspond
to how much future reward the agent predicts itself receiving at any given
point.

### Value Loss

These values will increase as the reward increases, and then should decrease
once reward becomes stable.
