# Training with Proximal Policy Optimization

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
