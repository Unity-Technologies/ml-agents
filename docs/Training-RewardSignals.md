# Reward Signals

In reinforcement learning, the end goal for the Agent is to discover a behavior (a Policy)
that maximizes a reward. Typically, a reward is defined by your environment, and corresponds 
to reaching some goal. These are what we refer to as "extrinsic" rewards, as they are defined
external of the learning algorithm. 

Rewards, however, can be defined outside of the enviroment as well, to encourage the agent to 
behave in certain ways, or to aid the learning of the true extrinsic reward. We refer to these
rewards as "intrinsic" reward signals. The total reward that the agent attempts to maximize can 
be a mix of extrinsic and intrinsic reward signals. 

ML-Agents allows reward signals to be defined in a modular way, and we provide three reward 
signals that can the mixed and matched to help shape your agent's behavior. The `extrinsic` Reward Signal represents the rewards defined in your environment, and is enabled by default. 
The `curiosity` reward signal helps your agent explore when extrinsic rewards are sparse.

## Enabling Reward Signals 

Reward signals, like other hyperparameters, are defined in the trainer config `.yaml` file. An
example is provided in `config/trainer_config.yaml`. To enable a reward signal, add it to the 
`reward_signals:` section under the brain name. For instance, to enable the extrinsic signal
in addition to a small curiosity reward, you would define your `reward_signals` as follows:

```
reward_signals:
    extrinsic:
        strength: 1.0
        gamma: 0.99
    curiosity:
        strength: 0.01
        gamma: 0.99
        encoding_size: 128
```

Each reward signal should define at least two parameters, `strength` and `gamma`, in addition 
to any class-specific hyperparameters. Note that to remove a reward signal, you should delete 
its entry entirely from `reward_signals`. At least one reward signal should be left defined
at all times. 

## Reward Signal Types

### The Extrinsic Reward Signal

The `extrinsic` reward signal is simply the reward given by the
[environment](Learning-Environment-Design.md). Remove it to force the agent
to ignore the environment reward. 

#### Strength 

`strength` is the factor by which to multiply the raw 
reward. Typical ranges will vary depending on the reward signal. 

Typical Range: `0.01 - `1.0`

#### Gamma

`gamma` corresponds to the discount factor for future rewards. This can be
thought of as how far into the future the agent should care about possible
rewards. In situations when the agent should be acting in the present in order
to prepare for rewards in the distant future, this value should be large. In
cases when rewards are more immediate, it can be smaller.

Typical Range: `0.8` - `0.995`

### The Curiosity Reward Signal

@chriselion

#### Strength 

In this case, `strength` corresponds to the magnitude of the curiosity reward generated 
by the intrinsic curiosity module. This should be scaled in order to ensure it is large enough 
to not be overwhelmed by extrinsic reward signals in the environment. 
Likewise it should not be too large to overwhelm the extrinsic reward signal.

Typical Range: `0.1 - `0.001`

#### Gamma

`gamma` corresponds to the discount factor for future rewards. 

Typical Range: `0.8` - `0.9`

#### Encoding Size

`encoding_size` corresponds to the size of the encoding used by the intrinsic curiosity model. 
This value should be small enough to encourage the ICM to compress the original
observation, but also not too small to prevent it from learning to differentiate between 
demonstrated and actual behavior. 

Default Value: 64
Typical Range: `64` - `256`

#### Learning Rate

`learning_rate` is the learning rate used to update the discriminator. 
This should typically be decreased if training is unstable, and the GAIL loss is unstable.

Default Value: `3e-4`
Typical Range: `1e-5` - `1e-3`  
