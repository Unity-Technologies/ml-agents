# Training with Proximal Policy Optimization

## Training Statistics

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
