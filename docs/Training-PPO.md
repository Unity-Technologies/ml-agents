# Training with Proximal Policy Optimization

ML-Agents provides an implementation of a reinforcement learning algorithm called
[Proximal Policy Optimization (PPO)](https://blog.openai.com/openai-baselines-ppo/).
PPO uses a neural network to approximate the ideal function that maps an agent's
observations to the best action an agent can take in a given state. The
ML-Agents PPO algorithm is implemented in TensorFlow and runs in a separate
Python process (communicating with the running Unity application over a socket).

ML-Agents also provides an implementation of
[Soft Actor-Critic (SAC)](https://bair.berkeley.edu/blog/2018/12/14/sac/). SAC tends
to be more _sample-efficient_, i.e. require fewer environment steps,
than PPO, but may spend more time performing model updates. This can produce a large
speedup on heavy or slow environments. Check out how to train with
SAC [here](Training-SAC.md).

To train an agent, you will need to provide the agent one or more reward signals which
the agent should attempt to maximize. See [Reward Signals](Reward-Signals.md)
for the available reward signals and the corresponding hyperparameters.

See [Training ML-Agents](Training-ML-Agents.md) for instructions on running the
training program, `learn.py`.

If you are using the recurrent neural network (RNN) to utilize memory, see
[Using Recurrent Neural Networks](Feature-Memory.md) for RNN-specific training
details.

If you are using curriculum training to pace the difficulty of the learning task
presented to an agent, see [Training with Curriculum
Learning](Training-Curriculum-Learning.md).

For information about imitation learning from demonstrations, see
[Training with Imitation Learning](Training-Imitation-Learning.md).

## Best Practices Training with PPO

Successfully training a Reinforcement Learning model often involves tuning the
training hyperparameters. This guide contains some best practices for tuning the
training process when the default parameters don't seem to be giving the level
of performance you would like.

## Hyperparameters

### Reward Signals

In reinforcement learning, the goal is to learn a Policy that maximizes reward.
At a base level, the reward is given by the environment. However, we could imagine
rewarding the agent for various different behaviors. For instance, we could reward
the agent for exploring new states, rather than just when an explicit reward is given.
Furthermore, we could mix reward signals to help the learning process.

Using `reward_signals` allows you to define [reward signals.](Reward-Signals.md)
The ML-Agents toolkit provides three reward signals by default, the Extrinsic (environment)
reward signal, the Curiosity reward signal, which can be used to encourage exploration in
sparse extrinsic reward environments, and the GAIL reward signal. Please see [Reward Signals](Reward-Signals.md)
for additional details.

### Lambda

`lambd` corresponds to the `lambda` parameter used when calculating the
Generalized Advantage Estimate ([GAE](https://arxiv.org/abs/1506.02438)). This
can be thought of as how much the agent relies on its current value estimate
when calculating an updated value estimate. Low values correspond to relying
more on the current value estimate (which can be high bias), and high values
correspond to relying more on the actual rewards received in the environment
(which can be high variance). The parameter provides a trade-off between the
two, and the right value can lead to a more stable training process.

Typical Range: `0.9` - `0.95`

### Buffer Size

`buffer_size` corresponds to how many experiences (agent observations, actions
and rewards obtained) should be collected before we do any learning or updating
of the model. **This should be a multiple of `batch_size`**. Typically a larger
`buffer_size` corresponds to more stable training updates.

Typical Range: `2048` - `409600`

### Batch Size

`batch_size` is the number of experiences used for one iteration of a gradient
descent update. **This should always be a fraction of the `buffer_size`**. If
you are using a continuous action space, this value should be large (in the
order of 1000s). If you are using a discrete action space, this value should be
smaller (in order of 10s).

Typical Range (Continuous): `512` - `5120`

Typical Range (Discrete): `32` - `512`

### Number of Epochs

`num_epoch` is the number of passes through the experience buffer during
gradient descent. The larger the `batch_size`, the larger it is acceptable to
make this. Decreasing this will ensure more stable updates, at the cost of
slower learning.

Typical Range: `3` - `10`

### Learning Rate

`learning_rate` corresponds to the strength of each gradient descent update
step. This should typically be decreased if training is unstable, and the reward
does not consistently increase.

Typical Range: `1e-5` - `1e-3`

### (Optional) Learning Rate Schedule

`learning_rate_schedule` corresponds to how the learning rate is changed over time.
For PPO, we recommend decaying learning rate until `max_steps` so learning converges
more stably. However, for some cases (e.g. training for an unknown amount of time)
this feature can be disabled.

Options:
* `linear` (default): Decay `learning_rate` linearly, reaching 0 at `max_steps`.
* `constant`: Keep learning rate constant for the entire training run.

Options: `linear`, `constant`

### Time Horizon

`time_horizon` corresponds to how many steps of experience to collect per-agent
before adding it to the experience buffer. When this limit is reached before the
end of an episode, a value estimate is used to predict the overall expected
reward from the agent's current state. As such, this parameter trades off
between a less biased, but higher variance estimate (long time horizon) and more
biased, but less varied estimate (short time horizon). In cases where there are
frequent rewards within an episode, or episodes are prohibitively large, a
smaller number can be more ideal. This number should be large enough to capture
all the important behavior within a sequence of an agent's actions.

Typical Range: `32` - `2048`

### Max Steps

`max_steps` corresponds to how many steps of the simulation (multiplied by
frame-skip) are run during the training process. This value should be increased
for more complex problems.

Typical Range: `5e5` - `1e7`

### Beta

`beta` corresponds to the strength of the entropy regularization, which makes
the policy "more random." This ensures that agents properly explore the action
space during training. Increasing this will ensure more random actions are
taken. This should be adjusted such that the entropy (measurable from
TensorBoard) slowly decreases alongside increases in reward. If entropy drops
too quickly, increase `beta`. If entropy drops too slowly, decrease `beta`.

Typical Range: `1e-4` - `1e-2`

### Epsilon

`epsilon` corresponds to the acceptable threshold of divergence between the old
and new policies during gradient descent updating. Setting this value small will
result in more stable updates, but will also slow the training process.

Typical Range: `0.1` - `0.3`

### Normalize

`normalize` corresponds to whether normalization is applied to the vector
observation inputs. This normalization is based on the running average and
variance of the vector observation. Normalization can be helpful in cases with
complex continuous control problems, but may be harmful with simpler discrete
control problems.

### Number of Layers

`num_layers` corresponds to how many hidden layers are present after the
observation input, or after the CNN encoding of the visual observation. For
simple problems, fewer layers are likely to train faster and more efficiently.
More layers may be necessary for more complex control problems.

Typical range: `1` - `3`

### Hidden Units

`hidden_units` correspond to how many units are in each fully connected layer of
the neural network. For simple problems where the correct action is a
straightforward combination of the observation inputs, this should be small. For
problems where the action is a very complex interaction between the observation
variables, this should be larger.

Typical Range: `32` - `512`

### (Optional) Visual Encoder Type

`vis_encode_type` corresponds to the encoder type for encoding visual observations.
Valid options include:
* `simple` (default): a simple encoder which consists of two convolutional layers
* `nature_cnn`: [CNN implementation proposed by Mnih et al.](https://www.nature.com/articles/nature14236),
consisting of three convolutional layers
* `resnet`: [IMPALA Resnet implementation](https://arxiv.org/abs/1802.01561),
consisting of three stacked layers, each with two residual blocks, making a
much larger network than the other two.

Options: `simple`, `nature_cnn`, `resnet`

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
used to store the hidden state of the recurrent neural network. This value must
be a multiple of 4, and should scale with the amount of information you expect
the agent will need to remember in order to successfully complete the task.

Typical Range: `64` - `512`

## (Optional) Behavioral Cloning Using Demonstrations

In some cases, you might want to bootstrap the agent's policy using behavior recorded
from a player. This can help guide the agent towards the reward. Behavioral Cloning (BC) adds
training operations that mimic a demonstration rather than attempting to maximize reward.

To use BC, add a `behavioral_cloning` section to the trainer_config. For instance:

```
    behavioral_cloning:
        demo_path: ./demos/ExpertPyramid.demo
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

## Training Statistics

To view training statistics, use TensorBoard. For information on launching and
using TensorBoard, see
[here](./Getting-Started-with-Balance-Ball.md#observing-training-progress).

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
