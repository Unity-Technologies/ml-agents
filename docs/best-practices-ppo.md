# Best Practices when training with PPO

The process of training a Reinforcement Learning model can often involve the need to tune the hyperparameters in order to achieve
a level of performance that is desirable. This guide contains some best practices for tuning the training process when the default
parameters don't seem to be giving the level of performance you would like.

## Hyperparameters

### Batch Size

`batch_size` corresponds to how many experiences are used for each gradient descent update. This should always be a fraction
of the `buffer_size`. If you are using a continuous action space, this value should be large (in 1000s). If you are using a discrete action space, this value should be smaller (in 10s). 

Typical Range (Continuous): `512` - `5120`

Typical Range (Discrete): `32` - `512`


### Beta (Used only in Discrete Control)

`beta` corresponds to the strength of the entropy regularization, which makes the policy "more random." This ensures that discrete action space agents properly explore during training. Increasing this will ensure more random actions are taken. This should be adjusted such that the entropy (measurable from TensorBoard) slowly decreases alongside increases in reward. If entropy drops too quickly, increase `beta`. If entropy drops too slowly, decrease `beta`.

Typical Range: `1e-4` - `1e-2`

### Buffer Size

`buffer_size` corresponds to how many experiences should be collected before gradient descent is performed on them all.
This should be a multiple of `batch_size`. Typically larger buffer sizes correspond to more stable training updates.

Typical Range: `2048` - `409600`

### Epsilon

`epsilon` corresponds to the acceptable threshold of divergence between the old and new policies during gradient descent updating. Setting this value small will result in more stable updates, but will also slow the training process.

Typical Range: `0.1` - `0.3`

### Hidden Units

`hidden_units` correspond to how many units are in each fully connected layer of the neural network. For simple problems
where the correct action is a straightforward combination of the state inputs, this should be small. For problems where
the action is a very complex interaction between the state variables, this should be larger.

Typical Range: `32` - `512`

### Learning Rate

`learning_rate` corresponds to the strength of each gradient descent update step. This should typically be decreased if
training is unstable, and the reward does not consistently increase.

Typical Range: `1e-5` - `1e-3`

### Number of Epochs

`num_epoch` is the number of passes through the experience buffer during gradient descent. The larger the batch size, the
larger it is acceptable to make this. Decreasing this will ensure more stable updates, at the cost of slower learning.

Typical Range: `3` - `10`

### Time Horizon

`time_horizon` corresponds to how many steps of experience to collect per-agent before adding it to the experience buffer.
In cases where there are frequent rewards within an episode, or episodes are prohibitively large, this can be a smaller number. For most stable training however, this number should be large enough to capture all the important behavior within a sequence of an agent's actions.

Typical Range: `64` - `2048`

### Max Steps

`max_steps` corresponds to how many steps of the simulation (multiplied by frame-skip) are run durring the training process. This value should be increased for more complex problems.

Typical Range: `5e5 - 1e7`

### Normalize 

`normalize` corresponds to whether normalization is applied to the state inputs. This normalization is based on the running average and variance of the states.
Normalization can be helpful in cases with complex continuous control problems, but may be harmful with simpler discrete control problems.

### Number of Layers

`num_layers` corresponds to how many hidden layers are present after the state input, or after the CNN encoding of the observation. For simple problems,
fewer layers are likely to train faster and more efficiently. More layers may be necessary for more complex control problems.

Typical range: `1` - `3`

## Training Statistics

To view training statistics, use Tensorboard. For information on launching and using Tensorboard, see [here](./Getting-Started-with-Balance-Ball.md#observing-training-progress).

### Cumulative Reward

The general trend in reward should consistently increase over time. Small ups and downs are to be expected. Depending on the complexity of the task, a significant increase in reward may not present itself until millions of steps into the training process.

### Entropy

This corresponds to how random the decisions of a brain are. This should consistently decrease during training. If it decreases too soon or not at all, `beta` should be adjusted (when using discrete action space).

### Learning Rate

This will decrease over time on a linear schedule.

### Policy Loss

These values will oscillate with training.

### Value Estimate

These values should increase with the reward. They corresponds to how much future reward the agent predicts itself receiving at any given point.

### Value Loss

These values will increase as the reward increases, and should decrease when reward becomes stable.
