# Training with Soft-Actor Critic

In addition to [Proximal Policy Optimization (PPO)](Training-PPO.md), ML-Agents also provides
[Soft Actor-Critic](http://bair.berkeley.edu/blog/2018/12/14/sac/) to perform
reinforcement learning.

In contrast with PPO, SAC is _off-policy_, which means it can learn from experiences collected
at any time during the past. As experiences are collected, they are placed in an
experience replay buffer and randomly drawn during training. This makes SAC
significantly more sample-efficient, often requiring 5-10 times less samples to learn
the same task as PPO. However, SAC tends to require more model updates. SAC is a
good choice for heavier or slower environments (about 0.1 seconds per step or more).

SAC is also a "maximum entropy" algorithm, and enables exploration in an intrinsic way.
Read more about maximum entropy RL [here](https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/).

To train an agent, you will need to provide the agent one or more reward signals which
the agent should attempt to maximize. See [Reward Signals](Reward-Signals.md)
for the available reward signals and the corresponding hyperparameters.

## Best Practices when training with SAC

Successfully training a reinforcement learning model often involves tuning
hyperparameters. This guide contains some best practices for training
when the default parameters don't seem to be giving the level of performance
you would like.

## Hyperparameters

### Reward Signals

In reinforcement learning, the goal is to learn a Policy that maximizes reward.
In the most basic case, the reward is given by the environment. However, we could imagine
rewarding the agent for various different behaviors. For instance, we could reward
the agent for exploring new states, rather than explicitly defined reward signals.
Furthermore, we could mix reward signals to help the learning process.

`reward_signals` provides a section to define [reward signals.](Reward-Signals.md)
ML-Agents provides two reward signals by default, the Extrinsic (environment) reward, and the
Curiosity reward, which can be used to encourage exploration in sparse extrinsic reward
environments.

#### Number of Updates for Reward Signal (Optional)

`reward_signal_num_update` for the reward signals corresponds to the number of mini batches sampled
and used for updating the reward signals during each
update. By default, we update the reward signals once every time the main policy is updated.
However, to imitate the training procedure in certain imitation learning papers (e.g.
[Kostrikov et. al](http://arxiv.org/abs/1809.02925), [Blondé et. al](http://arxiv.org/abs/1809.02064)),
we may want to update the policy N times, then update the reward signal (GAIL) M times.
We can change `train_interval` and `num_update` of SAC to N, as well as `reward_signal_num_update`
under `reward_signals` to M to accomplish this. By default, `reward_signal_num_update` is set to
`num_update`.

Typical Range: `num_update`

### Buffer Size

`buffer_size` corresponds the maximum number of experiences (agent observations, actions
and rewards obtained) that can be stored in the experience replay buffer. This value should be
large, on the order of thousands of times longer than your episodes, so that SAC
can learn from old as well as new experiences. It should also be much larger than
`batch_size`.

Typical Range: `50000` - `1000000`

### Buffer Init Steps

`buffer_init_steps` is the number of experiences to prefill the buffer with before attempting training.
As the untrained policy is fairly random, prefilling the buffer with random actions is
useful for exploration. Typically, at least several episodes of experiences should be
prefilled.

Typical Range: `1000` - `10000`

### Batch Size

`batch_size` is the number of experiences used for one iteration of a gradient
descent update. If
you are using a continuous action space, this value should be large (in the
order of 1000s). If you are using a discrete action space, this value should be
smaller (in order of 10s).

Typical Range (Continuous): `128` - `1024`

Typical Range (Discrete): `32` - `512`

### Initial Entropy Coefficient

`init_entcoef` refers to the initial entropy coefficient set at the beginning of training. In
SAC, the agent is incentivized to make its actions entropic to facilitate better exploration.
The entropy coefficient weighs the true reward with a bonus entropy reward. The entropy
coefficient is [automatically adjusted](https://arxiv.org/abs/1812.05905) to a preset target
entropy, so the `init_entcoef` only corresponds to the starting value of the entropy bonus.
Increase `init_entcoef` to explore more in the beginning, decrease to converge to a solution faster.

Typical Range (Continuous): `0.5` - `1.0`

Typical Range (Discrete): `0.05` - `0.5`

### Train Interval

`train_interval` is the number of steps taken between each agent training event. Typically,
we can train after every step, but if your environment's steps are very small and very frequent,
there may not be any new interesting information between steps, and `train_interval` can be increased.

Typical Range: `1` - `5`

### Number of Updates

`num_update` corresponds to the number of mini batches sampled and used for training during each
training event. In SAC, a single "update" corresponds to grabbing a batch of size `batch_size` from the experience
replay buffer, and using this mini batch to update the models. Typically, this can be left at 1.
However, to imitate the training procedure in certain papers (e.g.
[Kostrikov et. al](http://arxiv.org/abs/1809.02925), [Blondé et. al](http://arxiv.org/abs/1809.02064)),
we may want to update N times with different mini batches before grabbing additional samples.
We can change `train_interval` and `num_update` to N to accomplish this.

Typical Range: `1`

### Tau

`tau` corresponds to the magnitude of the target Q update during the SAC model update.
In SAC, there are two neural networks: the target and the policy. The target network is
used to bootstrap the policy's estimate of the future rewards at a given state, and is fixed
while the policy is being updated. This target is then slowly updated according to `tau`.
Typically, this value should be left at `0.005`. For simple problems, increasing
`tau` to `0.01` might reduce the time it takes to learn, at the cost of stability.

Typical Range: `0.005` - `0.01`

### Learning Rate

`learning_rate` corresponds to the strength of each gradient descent update
step. This should typically be decreased if training is unstable, and the reward
does not consistently increase.

Typical Range: `1e-5` - `1e-3`

### (Optional) Learning Rate Schedule

`learning_rate_schedule` corresponds to how the learning rate is changed over time.
For SAC, we recommend holding learning rate constant so that the agent can continue to
learn until its Q function converges naturally.

Options:
* `linear`: Decay `learning_rate` linearly, reaching 0 at `max_steps`.
* `constant` (default): Keep learning rate constant for the entire training run.

Options: `linear`, `constant`

### Time Horizon

`time_horizon` corresponds to how many steps of experience to collect per-agent
before adding it to the experience buffer. This parameter is a lot less critical
to SAC than PPO, and can typically be set to approximately your episode length.

Typical Range: `32` - `2048`

### Max Steps

`max_steps` corresponds to how many steps of the simulation (multiplied by
frame-skip) are run during the training process. This value should be increased
for more complex problems.

Typical Range: `5e5` - `1e7`

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

### (Optional) Save Replay Buffer

`save_replay_buffer` enables you to save and load the experience replay buffer as well as
the model when quitting and re-starting training. This may help resumes go more smoothly,
as the experiences collected won't be wiped. Note that replay buffers can be very large, and
will take up a considerable amount of disk space. For that reason, we disable this feature by
default.

Default: `False`

## (Optional) Pretraining Using Demonstrations

In some cases, you might want to bootstrap the agent's policy using behavior recorded
from a player. This can help guide the agent towards the reward. Pretraining adds
training operations that mimic a demonstration rather than attempting to maximize reward.
It is essentially equivalent to running [behavioral cloning](./Training-Behavioral-Cloning.md)
in-line with SAC.

To use pretraining, add a `pretraining` section to the trainer_config. For instance:

```
    pretraining:
        demo_path: ./demos/ExpertPyramid.demo
        strength: 0.5
        steps: 10000
```

Below are the available hyperparameters for pretraining.

### Strength

`strength` corresponds to the learning rate of the imitation relative to the learning
rate of SAC, and roughly corresponds to how strongly we allow the behavioral cloning
to influence the policy.

Typical Range: `0.1` - `0.5`

### Demo Path

`demo_path` is the path to your `.demo` file or directory of `.demo` files.
See the [imitation learning guide](Training-Imitation-Learning.md) for more on `.demo` files.

### Steps

During pretraining, it is often desirable to stop using demonstrations after the agent has
"seen" rewards, and allow it to optimize past the available demonstrations and/or generalize
outside of the provided demonstrations. `steps` corresponds to the training steps over which
pretraining is active. The learning rate of the pretrainer will anneal over the steps. Set
the steps to 0 for constant imitation over the entire training run.

### (Optional) Batch Size

`batch_size` is the number of demonstration experiences used for one iteration of a gradient
descent update. If not specified, it will default to the `batch_size` defined for SAC.

Typical Range (Continuous): `512` - `5120`

Typical Range (Discrete): `32` - `512`

## Training Statistics

To view training statistics, use TensorBoard. For information on launching and
using TensorBoard, see
[here](./Getting-Started-with-Balance-Ball.md#observing-training-progress).

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
