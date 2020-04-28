# Training with Soft-Actor Critic

## Training Statistics

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
