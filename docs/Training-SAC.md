# Training with Soft-Actor Critic

To train an agent, you will need to provide the agent one or more reward signals which
the agent should attempt to maximize. See [Reward Signals](Reward-Signals.md)
for the available reward signals and the corresponding hyperparameters.

## Best Practices when training with SAC

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

#### Steps Per Update for Reward Signal (Optional)

`reward_signal_steps_per_update` for the reward signals corresponds to the number of steps per mini batch sampled
and used for updating the reward signals. By default, we update the reward signals once every time the main policy is updated.
However, to imitate the training procedure in certain imitation learning papers (e.g.
[Kostrikov et. al](http://arxiv.org/abs/1809.02925), [Blondé et. al](http://arxiv.org/abs/1809.02064)),
we may want to update the reward signal (GAIL) M times for every update of the policy.
We can change `steps_per_update` of SAC to N, as well as `reward_signal_steps_per_update`
under `reward_signals` to N / M to accomplish this. By default, `reward_signal_steps_per_update` is set to
`steps_per_update`.

Typical Range: `steps_per_update`


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
rate of SAC, and roughly corresponds to how strongly we allow BC
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
descent update. If not specified, it will default to the `batch_size` defined for SAC.

Typical Range (Continuous): `512` - `5120`

Typical Range (Discrete): `32` - `512`


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
