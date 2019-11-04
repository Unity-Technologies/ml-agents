# Reward Signals

In reinforcement learning, the end goal for the Agent is to discover a behavior (a Policy)
that maximizes a reward. Typically, a reward is defined by your environment, and corresponds
to reaching some goal. These are what we refer to as "extrinsic" rewards, as they are defined
external of the learning algorithm.

Rewards, however, can be defined outside of the environment as well, to encourage the agent to
behave in certain ways, or to aid the learning of the true extrinsic reward. We refer to these
rewards as "intrinsic" reward signals. The total reward that the agent will learn to maximize can
be a mix of extrinsic and intrinsic reward signals.

ML-Agents allows reward signals to be defined in a modular way, and we provide three reward
signals that can the mixed and matched to help shape your agent's behavior. The `extrinsic` Reward
Signal represents the rewards defined in your environment, and is enabled by default.
The `curiosity` reward signal helps your agent explore when extrinsic rewards are sparse.

## Enabling Reward Signals

Reward signals, like other hyperparameters, are defined in the trainer config `.yaml` file. An
example is provided in `config/trainer_config.yaml` and `config/gail_config.yaml`. To enable a reward signal, add it to the
`reward_signals:` section under the behavior name. For instance, to enable the extrinsic signal
in addition to a small curiosity reward and a GAIL reward signal, you would define your `reward_signals` as follows:

```yaml
reward_signals:
    extrinsic:
        strength: 1.0
        gamma: 0.99
    curiosity:
        strength: 0.02
        gamma: 0.99
        encoding_size: 256
    gail:
        strength: 0.01
        gamma: 0.99
        encoding_size: 128
        demo_path: demos/ExpertPyramid.demo
```

Each reward signal should define at least two parameters, `strength` and `gamma`, in addition
to any class-specific hyperparameters. Note that to remove a reward signal, you should delete
its entry entirely from `reward_signals`. At least one reward signal should be left defined
at all times.

## Reward Signal Types
As part of the toolkit, we provide three reward signal types as part of hyperparameters - Extrinsic, Curiosity, and GAIL.

### Extrinsic Reward Signal

The `extrinsic` reward signal is simply the reward given by the
[environment](Learning-Environment-Design.md). Remove it to force the agent
to ignore the environment reward.

#### Strength

`strength` is the factor by which to multiply the raw
reward. Typical ranges will vary depending on the reward signal.

Typical Range: `1.0`

#### Gamma

`gamma` corresponds to the discount factor for future rewards. This can be
thought of as how far into the future the agent should care about possible
rewards. In situations when the agent should be acting in the present in order
to prepare for rewards in the distant future, this value should be large. In
cases when rewards are more immediate, it can be smaller.

Typical Range: `0.8` - `0.995`

### Curiosity Reward Signal

The `curiosity` Reward Signal enables the Intrinsic Curiosity Module. This is an implementation
of the approach described in "Curiosity-driven Exploration by Self-supervised Prediction"
by Pathak, et al. It trains two networks:
* an inverse model, which takes the current and next observation of the agent, encodes them, and
uses the encoding to predict the action that was taken between the observations
* a forward model, which takes the encoded current observation and action, and predicts the
next encoded observation.

The loss of the forward model (the difference between the predicted and actual encoded observations) is used as the intrinsic reward, so the more surprised the model is, the larger the reward will be.

For more information, see
* https://arxiv.org/abs/1705.05363
* https://pathak22.github.io/noreward-rl/
* https://blogs.unity3d.com/2018/06/26/solving-sparse-reward-tasks-with-curiosity/

#### Strength

In this case, `strength` corresponds to the magnitude of the curiosity reward generated
by the intrinsic curiosity module. This should be scaled in order to ensure it is large enough
to not be overwhelmed by extrinsic reward signals in the environment.
Likewise it should not be too large to overwhelm the extrinsic reward signal.

Typical Range: `0.001` - `0.1`

#### Gamma

`gamma` corresponds to the discount factor for future rewards.

Typical Range: `0.8` - `0.995`

#### (Optional) Encoding Size

`encoding_size` corresponds to the size of the encoding used by the intrinsic curiosity model.
This value should be small enough to encourage the ICM to compress the original
observation, but also not too small to prevent it from learning to differentiate between
demonstrated and actual behavior.

Default Value: `64`

Typical Range: `64` - `256`

#### (Optional) Learning Rate

`learning_rate` is the learning rate used to update the intrinsic curiosity module.
This should typically be decreased if training is unstable, and the curiosity loss is unstable.

Default Value: `3e-4`

Typical Range: `1e-5` - `1e-3`

### GAIL Reward Signal

GAIL, or [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), is an
imitation learning algorithm that uses an adversarial approach, in a similar vein to GANs
(Generative Adversarial Networks). In this framework, a second neural network, the
discriminator, is taught to distinguish whether an observation/action is from a demonstration or
produced by the agent. This discriminator can the examine a new observation/action and provide it a
reward based on how close it believes this new observation/action is to the provided demonstrations.

At each training step, the agent tries to learn how to maximize this reward. Then, the
discriminator is trained to better distinguish between demonstrations and agent state/actions.
In this way, while the agent gets better and better at mimicing the demonstrations, the
discriminator keeps getting stricter and stricter and the agent must try harder to "fool" it.

This approach, when compared to [Behavioral Cloning](Training-Behavioral-Cloning.md), requires
far fewer demonstrations to be provided. After all, we are still learning a policy that happens
to be similar to the demonstrations, not directly copying the behavior of the demonstrations. It
is especially effective when combined with an Extrinsic signal. However, the GAIL reward signal can
also be used independently to purely learn from demonstrations.

Using GAIL requires recorded demonstrations from your Unity environment. See the
[imitation learning guide](Training-Imitation-Learning.md) to learn more about recording demonstrations.

#### Strength

`strength` is the factor by which to multiply the raw reward. Note that when using GAIL
with an Extrinsic Signal, this value should be set lower if your demonstrations are
suboptimal (e.g. from a human), so that a trained agent will focus on receiving extrinsic
rewards instead of exactly copying the demonstrations. Keep the strength below about 0.1 in those cases.

Typical Range: `0.01` - `1.0`

#### Gamma

`gamma` corresponds to the discount factor for future rewards.

Typical Range: `0.8` - `0.9`

#### Demo Path

`demo_path` is the path to your `.demo` file or directory of `.demo` files. See the [imitation learning guide]
(Training-Imitation-Learning.md).

#### (Optional) Encoding Size

`encoding_size` corresponds to the size of the hidden layer used by the discriminator.
This value should be small enough to encourage the discriminator to compress the original
observation, but also not too small to prevent it from learning to differentiate between
demonstrated and actual behavior. Dramatically increasing this size will also negatively affect
training times.

Default Value: `64`

Typical Range: `64` - `256`

#### (Optional) Learning Rate

`learning_rate` is the learning rate used to update the discriminator.
This should typically be decreased if training is unstable, and the GAIL loss is unstable.

Default Value: `3e-4`

Typical Range: `1e-5` - `1e-3`

#### (Optional) Use Actions

`use_actions` determines whether the discriminator should discriminate based on both
observations and actions, or just observations. Set to `True` if you want the agent to
mimic the actions from the demonstrations, and `False` if you'd rather have the agent
visit the same states as in the demonstrations but with possibly different actions.
Setting to `False` is more likely to be stable, especially with imperfect demonstrations,
but may learn slower.

Default Value: `false`

#### (Optional) Variational Discriminator Bottleneck

`use_vail` enables a [variational bottleneck](https://arxiv.org/abs/1810.00821) within the
GAIL discriminator. This forces the discriminator to learn a more general representation
and reduces its tendency to be "too good" at discriminating, making learning more stable.
However, it does increase training time. Enable this if you notice your imitation learning is
unstable, or unable to learn the task at hand.

Default Value: `false`
