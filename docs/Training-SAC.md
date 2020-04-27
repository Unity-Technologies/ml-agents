# Training with Soft-Actor Critic

## Best Practices when training with SAC


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
used to store the hidden state of the recurrent neural network in the policy.
This value must be a multiple of 2, and should scale with the amount of information you expect
the agent will need to remember in order to successfully complete the task.

Typical Range: `32` - `256`

## Training Statistics

To view training statistics, use TensorBoard. For information on launching and
using TensorBoard, see
[here](./Getting-Started.md#observing-training-progress).

### Cumulative Reward

The general trend in reward should consistently increase over time. Small ups
and downs are to be expected. Depending on the complexity of the task, a
significant increase in reward may not present itself until millions of steps
into the training process.

### Entropy Coefficient

SAC is a "maximum entropy" reinforcement learning algorithm, and agents trained using
SAC are incentivized to behave randomly while also solving the problem. The entropy
coefficient balances the incentive to behave randomly vs. maximizing the reward.
This value is adjusted automatically so that the agent retains some amount of randomness during
training. It should steadily decrease in the beginning of training, and reach some small
value where it will level off. If it decreases too soon or takes too
long to decrease, `init_entcoef` should be adjusted.

### Entropy

This corresponds to how random the decisions are. This should
initially increase during training, reach a peak, and should decline along
with the Entropy Coefficient. This is because in the beginning, the agent is
incentivized to be more random for exploration due to a high entropy coefficient.
If it decreases too soon or takes too long to decrease, `init_entcoef` should be adjusted.

### Learning Rate

This will stay a constant value by default, unless `learning_rate_schedule`
is set to `linear`.

### Policy Loss

These values may increase as the agent explores, but should decrease long-term
as the agent learns how to solve the task.

### Value Estimate

These values should increase as the cumulative reward increases. They correspond
to how much future reward the agent predicts itself receiving at any given
point. They may also increase at the beginning as the agent is rewarded for
being random (see: Entropy and Entropy Coefficient), but should decline as
Entropy Coefficient decreases.

### Value Loss

These values will increase as the reward increases, and then should decrease
once reward becomes stable.
