# Training ML-Agents

For a broad overview of reinforcement learning, imitation learning and all the
training scenarios, methods and options within the ML-Agents Toolkit, see
[ML-Agents Toolkit Overview](ML-Agents-Overview.md).

Once your learning environment has been created and is ready for training, the
next step is to initiate a training run. Training in the ML-Agents Toolkit is
powered by a dedicated Python package, `mlagents`. This package exposes a
command `mlagents-learn` that is the single entry point for all training
workflows (e.g. reinforcement leaning, imitation learning, curriculum learning).
Its implementation can be found at
[ml-agents/mlagents/trainers/learn.py](../ml-agents/mlagents/trainers/learn.py).

## Training with mlagents-learn

### Starting Training

`mlagents-learn` is the main training utility provided by the ML-Agents Toolkit.
It accepts a number of CLI options in addition to a YAML configuration file that
contains all the configurations and hyperparameters to be used during training.
The set of configurations and hyperparameters to include in this file depend on
the agents in your environment and the specific training method you wish to
utilize. Keep in mind that the hyperparameter values can have a big impact on
the training performance (i.e. your agent's ability to learn a policy that
solves the task). In this page, we will review all the hyperparameters for all
training methods and provide guidelines and advice on their values.

To view a description of all the CLI options accepted by `mlagents-learn`, use
the `--help`:

```sh
mlagents-learn --help
```

The basic command for training is:

```sh
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>
```

where

- `<trainer-config-file>` is the file path of the trainer configuration yaml.
  This contains all the hyperparameter values. We offer a detailed guide on the
  structure of this file and the meaning of the hyperparameters (and advice on how
  to set them) in the dedicated [Training Configurations](#training-configurations)
  section below.
- `<env_name>`**(Optional)** is the name (including path) of your
  [Unity executable](Learning-Environment-Executable.md) containing the agents
  to be trained. If `<env_name>` is not passed, the training will happen in the
  Editor. Press the :arrow_forward: button in Unity when the message _"Start
  training by pressing the Play button in the Unity Editor"_ is displayed on
  the screen.
- `<run-identifier>` is a unique name you can use to identify the results of
  your training runs.

See the
[Getting Started Guide](Getting-Started.md#training-a-new-model-with-reinforcement-learning)
for a sample execution of the `mlagents-learn` command.

#### Observing Training

Regardless of which training methods, configurations or hyperparameters you
provide, the training process will always generate three artifacts:

1. Summaries (under the `summaries/` folder): these are training metrics that
   are updated throughout the training process. They are helpful to monitor your
   training performance and may help inform how to update your hyperparameter
   values. See [Using TensorBoard](Using-Tensorboard.md) for more details on how
   to visualize the training metrics.
1. Models (under the `models/` folder): these contain the model checkpoints that
   are updated throughout training and the final model file (`.nn`). This final
   model file is generated once either when training completes or is
   interrupted.
1. Timers file (also under the `summaries/` folder): this contains aggregated
   metrics on your training process, including time spent on specific code
   blocks. See [Profiling in Python](Profiling-Python.md) for more information
   on the timers generated.

These artifacts (except the `.nn` file) are updated throughout the training
process and finalized when training completes or is interrupted.

#### Stopping and Resuming Training

To interrupt training and save the current progress, hit `Ctrl+C` once and wait
for the model(s) to be saved out.

To resume a previously interrupted or completed training run, use the `--resume`
flag and make sure to specify the previously used run ID.

If you would like to re-run a previously interrupted or completed training run
and re-use the same run ID (in this case, overwriting the previously generated
artifacts), then use the `--force` flag.

#### Loading an Existing Model

You can also use this mode to run inference of an already-trained model in
Python by using both the `--resume` and `--inference` flags. Note that if you
want to run inference in Unity, you should use the
[Unity Inference Engine](Getting-Started.md#running-a-pre-trained-model).

Alternatively, you might want to start a new training run but _initialize_ it
using an already-trained model. You may want to do this, for instance, if your
environment changed and you want a new model, but the old behavior is still
better than random. You can do this by specifying
`--initialize-from=<run-identifier>`, where `<run-identifier>` is the old run
ID.

## Training Configurations

The Unity ML-Agents Toolkit provides a wide range of training scenarios, methods
and options. As such, specific training runs may require different training
configurations and may generate different artifacts and TensorBoard statistics.
This section offers a detailed guide into how to manage the different training
set-ups withing the toolkit.

More specifically, this section offers a detailed guide on four command-line
flags for `mlagents-learn` that control the training configurations:

- `<trainer-config-file>`: defines the training hyperparameters for each
  Behavior in the scene
- `--curriculum`: defines the set-up for Curriculum Learning
- `--sampler`: defines the set-up for Environment Parameter Randomization
- `--num-envs`: number of concurrent Unity instances to use during training

Reminder that a detailed description of all command-line options can be found by
using the help utility:

```sh
mlagents-learn --help
```

It is important to highlight that successfully training a Behavior in the
ML-Agents Toolkit involves tuning the training hyperparameters and
configuration. This guide contains some best practices for tuning the training
process when the default parameters don't seem to be giving the level of
performance you would like. We provide sample configuration files for our
example environments in the [config/](../config/) directory. The
`config/trainer_config.yaml` was used to train the 3D Balance Ball in the
[Getting Started](Getting-Started.md) guide. That configuration file uses the
PPO trainer, but we also have configuration files for SAC and GAIL.

Additionally, the set of configurations you provide depend on the training
functionalities you use (see [ML-Agents Toolkit Overview](ML-Agents-Overview.md)
for a description of all the training functionalities). Each functionality you
add typically has its own training configurations or additional configuration
files. For instance:

- Use PPO or SAC?
- Use Recurrent Neural Networks for adding memory to your agents?
- Use the intrinsic curiosity module?
- Ignore the environment reward signal?
- Pre-train using behavioral cloning? (Assuming you have recorded
  demonstrations.)
- Include the GAIL intrinsic reward signals? (Assuming you have recorded
  demonstrations.)
- Use self-play? (Assuming your environment includes multiple agents.)

The answers to the above questions will dictate the configuration files and the
parameters within them. The rest of this section breaks down the different
configuration files and explains the possible settings for each.

### Trainer Config File

We begin with the trainer config file, `<trainer-config-file>`, which includes
a set of configurations for each Behavior in your scene. Some of the
configurations are required while others are optional. To help us get started,
below is a sample file that includes all the possible settings if we're using a
PPO trainer with all the possible training functionalities enabled (memory,
behavioral cloning, curiosity, GAIL and self-play). You will notice that
curriculum and environment parameter randomization settings are not part of
this file, but their settings live in different files that we'll cover in
subsequent sections.

```yaml
BehaviorPPO:
  trainer: ppo

  # Trainer configs common to PPO/SAC (excluding reward signals)
  batch_size: 1024
  buffer_size: 10240
  hidden_units: 128
  learning_rate: 3.0e-4
  learning_rate_schedule: linear
  max_steps: 5.0e5
  normalize: false
  num_layers: 2
  time_horizon: 64
  vis_encoder_type: simple

  # PPO-specific configs
  beta: 5.0e-3
  epsilon: 0.2
  lambd: 0.95
  num_epoch: 3

  # memory
  use_recurrent: true
  sequence_length: 64
  memory_size: 256

  # behavior cloning
  behavioral_cloning:
    demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
    strength: 0.5
    steps: 150000
    batch_size: 512
    num_epoch: 3
    samples_per_update: 0
    init_path:

  reward_signals:
    # environment reward
    extrinsic:
      strength: 1.0
      gamma: 0.99

    # curiosity module
    curiosity:
      strength: 0.02
      gamma: 0.99
      encoding_size: 256
      learning_rate: 3e-4

    # GAIL
    gail:
      strength: 0.01
      gamma: 0.99
      encoding_size: 128
      demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
      learning_rate: 3e-4
      use_actions: false
      use_vail: false

  # self-play
  self_play:
    window: 10
    play_against_latest_model_ratio: 0.5
    save_steps: 50000
    swap_steps: 50000
    team_change: 100000
```

Here is an equivalent file if we use an SAC trainer instead. Notice that the
configurations for the additional functionalities (memory, behavioral cloning,
curiosity and self-play) remain unchanged.

```yaml
BehaviorSAC:
  trainer: sac

  # Trainer configs common to PPO/SAC (excluding reward signals)
  # same as PPO config

  # SAC-specific configs (replaces the "PPO-specific configs" section above)
  buffer_init_steps: 0
  tau: 0.005
  steps_per_update: 1
  train_interval: 1
  init_entcoef: 1.0
  save_replay_buffer: false

  # memory
  # same as PPO config

  # pre-training using behavior cloning
  behavioral_cloning:
    # same as PPO config

  reward_signals:
    reward_signal_num_update: 1 # only applies to SAC

    # environment reward
    extrinsic:
      # same as PPO config

    # curiosity module
    curiosity:
      # same as PPO config

    # GAIL
    gail:
      # same as PPO config

  # self-play
  self_play:
    # same as PPO config
```

We now break apart the components of the configuration file and describe what
each of these parameters mean and provide guidelines on how to set them.

#### Common Trainer Configurations

One of the first decisions you need to make regarding your training run is which
trainer to use: PPO or SAC. There are some training configurations that are
common to both trainers (which we review now) and others that depend on the
choice of the trainer (which we review on subsequent sections).

| **Setting**            |  **Description** |
| :--------------------- | :-------------- |
| `trainer`                | The type of training to perform: `ppo` or `sac` |
| `init_path`              | Initialize trainer from a previously saved model. Note that the prior run should have used the same trainer configurations as the current run, and have been saved with the same version of ML-Agents. <br><br>You should provide the full path to the folder where the checkpoints were saved, e.g. `./models/{run-id}/{behavior_name}`. This option is provided in case you want to initialize different behaviors from different runs; in most cases, it is sufficient to use the `--initialize-from` CLI parameter to initialize all models from the same run.|
| `summary_freq`           | Number of experiences that needs to be collected before generating and displaying training statistics. This determines the granularity of the graphs in Tensorboard.             |
| `batch_size`             | Number of experiences in each iteration of gradient descent. **This should always be a fraction of the `buffer_size`**. If you are using a continuous action space, this value should be large (in the order of 1000s). If you are using a discrete action space, this value should be smaller (in order of 10s). <br><br> Typical range: (Continuous - PPO): `512` - `5120`; (Continuous - SAC): `128` - `1024`; (Discrete, PPO & SAC): `32` - `512`. |
| `buffer_size`            | Number of experiences to collect before updating the policy model. Corresponds to how many experiences (agent observations, actions and rewards obtained) should be collected before we do any learning or updating of the model. **This should be a multiple of `batch_size`**. Typically a larger `buffer_size` corresponds to more stable training updates. In SAC, the max size of the experience buffer - on the order of thousands of times longer than your episodes, so that SAC can learn from old as well as new experiences. <br><br>Typical range: PPO: `2048` - `409600`; SAC: `50000` - `1000000`|
| `hidden_units`           | Number of units in the hidden layers of the neural network. Correspond to how many units are in each fully connected layer of the neural network. For simple problems where the correct action is a straightforward combination of the observation inputs, this should be small. For problems where the action is a very complex interaction between the observation variables, this should be larger. <br><br> Typical range: `32` - `512`|
| `learning_rate`          | Initial learning rate for gradient descent. Corresponds to the strength of each gradient descent update step. This should typically be decreased if training is unstable, and the reward does not consistently increase. <br><br>Typical range: `1e-5` - `1e-3` |
| `learning_rate_schedule` | Determines how learning rate changes over time. For PPO, we recommend decaying learning rate until max_steps so learning converges more stably. However, for some cases (e.g. training for an unknown amount of time) this feature can be disabled. For SAC, we recommend holding learning rate constant so that the agent can continue to learn until its Q function converges naturally. <br><br>`linear` (default) decays the learning_rate linearly, reaching 0 at max_steps, while `constant` keeps the learning rate constant for the entire training run.  |
| `max_steps`              | Total number of experience points that must be collected from the simulation before ending the training process. <br><br>Typical range: `5e5` - `1e7`|
| `normalize`              | Whether normalization is applied to the vector observation inputs. This normalization is based on the running average and variance of the vector observation. Normalization can be helpful in cases with complex continuous control problems, but may be harmful with simpler discrete control problems. |
| `num_layers`             | The number of hidden layers in the neural network. Corresponds to how many hidden layers are present after the observation input, or after the CNN encoding of the visual observation. For simple problems, fewer layers are likely to train faster and more efficiently. More layers may be necessary for more complex control problems. <br><br> Typical range: `1` - `3` |
| `time_horizon`           | How many steps of experience to collect per-agent before adding it to the experience buffer. When this limit is reached before the end of an episode, a value estimate is used to predict the overall expected reward from the agent's current state. As such, this parameter trades off between a less biased, but higher variance estimate (long time horizon) and more biased, but less varied estimate (short time horizon). In cases where there are frequent rewards within an episode, or episodes are prohibitively large, a smaller number can be more ideal. This number should be large enough to capture all the important behavior within a sequence of an agent's actions. <br><br> Typical range: `32` - `2048` |
| `vis_encoder_type`       | Encoder type for encoding visual observations. <br><br> `simple` (default) uses a simple encoder which consists of two convolutional layers, `nature_cnn` uses the CNN implementation proposed by [Mnih et al.](https://www.nature.com/articles/nature14236), consisting of three convolutional layers, and `resnet` uses the [IMPALA Resnet](https://arxiv.org/abs/1802.01561) consisting of three stacked layers, each with two residual blocks, making a much larger network than the other two. |

#### Trainer-specific Configurations

Depending on your choice of a trainer, there are additional trainer-specific
configurations. We present them below in two separate tables, but keep in mind
that you only need to include the configurations for the trainer selected (i.e.
the `trainer` setting above).

PPO-specific configurations:

| **Setting** | **Description** |
| :---------- | :--------------- |
| `beta`        | Strength of the entropy regularization, which makes the policy "more random." This ensures that agents properly explore the action space during training. Increasing this will ensure more random actions are taken. This should be adjusted such that the entropy (measurable from TensorBoard) slowly decreases alongside increases in reward. If entropy drops too quickly, increase beta. If entropy drops too slowly, decrease `beta`. <br><br>Typical range: `1e-4` - `1e-2` |
| `epsilon`     | Influences how rapidly the policy can evolve during training. Corresponds to the acceptable threshold of divergence between the old and new policies during gradient descent updating. Setting this value small will result in more stable updates, but will also slow the training process. <br><br>Typical range: `0.1` - `0.3` |
| `lambd`       | Regularization parameter (lambda) used when calculating the Generalized Advantage Estimate ([GAE](https://arxiv.org/abs/1506.02438)). This can be thought of as how much the agent relies on its current value estimate when calculating an updated value estimate. Low values correspond to relying more on the current value estimate (which can be high bias), and high values correspond to relying more on the actual rewards received in the environment (which can be high variance). The parameter provides a trade-off between the two, and the right value can lead to a more stable training process. <br><br>Typical range: `0.9` - `0.95` |
| `num_epoch`   | Number of passes to make through the experience buffer when performing gradient descent optimization.The larger the batch_size, the larger it is acceptable to make this. Decreasing this will ensure more stable updates, at the cost of slower learning. <br><br>Typical range: `3` - `10` |

SAC-specific configurations:

| **Setting**        | **Description** |
| :------------------| :--------------- |
| `buffer_init_steps`  | Number of experiences to collect into the buffer before updating the policy model. As the untrained policy is fairly random, pre-filling the buffer with random actions is useful for exploration. Typically, at least several episodes of experiences should be pre-filled. <br><br>Typical range: `1000` - `10000` |
| `init_entcoef`       | How much the agent should explore in the beginning of training. Corresponds to the initial entropy coefficient set at the beginning of training. In SAC, the agent is incentivized to make its actions entropic to facilitate better exploration. The entropy coefficient weighs the true reward with a bonus entropy reward. The entropy coefficient is [automatically adjusted](https://arxiv.org/abs/1812.05905) to a preset target entropy, so the `init_entcoef` only corresponds to the starting value of the entropy bonus. Increase init_entcoef to explore more in the beginning, decrease to converge to a solution faster. <br><br>Typical range: (Continuous): `0.5` - `1.0`; (Discrete): `0.05` - `0.5` |
| `save_replay_buffer` | (Optional, default = `false`) Whether to save and load the experience replay buffer as well as the model when quitting and re-starting training. This may help resumes go more smoothly, as the experiences collected won't be wiped. Note that replay buffers can be very large, and will take up a considerable amount of disk space. For that reason, we disable this feature by default. |
| `tau`                | How aggressively to update the target network used for bootstrapping value estimation in SAC. Corresponds to the magnitude of the target Q update during the SAC model update. In SAC, there are two neural networks: the target and the policy. The target network is used to bootstrap the policy's estimate of the future rewards at a given state, and is fixed while the policy is being updated. This target is then slowly updated according to tau. Typically, this value should be left at 0.005. For simple problems, increasing tau to 0.01 might reduce the time it takes to learn, at the cost of stability. <br><br>Typical range: `0.005` - `0.01` |
| `steps_per_update`     | Average ratio of agent steps (actions) taken to updates made of the agent's policy. In SAC, a single "update" corresponds to grabbing a batch of size `batch_size` from the experience replay buffer, and using this mini batch to update the models. Note that it is not guaranteed that after exactly `steps_per_update` steps an update will be made, only that the ratio will hold true over many steps. Typically, `steps_per_update` should be greater than or equal to 1. Note that setting `steps_per_update` lower will improve sample efficiency (reduce the number of steps required to train) but increase the CPU time spent performing updates. For most environments where steps are fairly fast (e.g. our example environments) `steps_per_update` equal to the number of agents in the scene is a good balance. For slow environments (steps take 0.1 seconds or more) reducing `steps_per_update` may improve training speed. We can also change `steps_per_update` to lower than 1 to update more often than once per step, though this will usually result in a slowdown unless the environment is very slow. <br><br>Typical range: `1` - `20` |
| `train_interval` | Number of steps taken between each agent training event. Typically, we can train after every step, but if your environment's steps are very small and very frequent, there may not be any new interesting information between steps, and `train_interval` can be increased. <br><br>Typical range: `1` - `5` |

#### Reward Signals

The `reward_signals` section enables the specification of settings for both
extrinsic (i.e. environment-based) and intrinsic reward signals (e.g. curiosity
and GAIL). Each reward signal should define at least two parameters, `strength`
and `gamma`, in addition to any class-specific hyperparameters. Note that to
remove a reward signal, you should delete its entry entirely from
`reward_signals`. At least one reward signal should be left defined at all
times. Provide the following configurations to design the reward signal for
your training run:

**Extrinsic rewards** - Enable these settings to ensure that your training run
incorporates your environment-based reward signal:

| **Setting**        | **Description** |
| :------------------| :--------------- |
| `extrinsic > strength` | Factor by which to multiply the reward given by the environment. Typical ranges will vary depending on the reward signal. <br><br>Typical range: `1.00` |
| `extrinsic > gamma`    | Discount factor for future rewards coming from the environment. This can be thought of as how far into the future the agent should care about possible rewards. In situations when the agent should be acting in the present in order to prepare for rewards in the distant future, this value should be large. In cases when rewards are more immediate, it can be smaller. Must be strictly smaller than 1. <br><br>Typical range: `0.8` - `0.995` |

**Curiosity intrinsic reward**- To enable curiosity, provide these settings:

| **Setting**        | **Description** |
| :------------------| :--------------- |
| `curiosity > strength`      | Magnitude of the curiosity reward generated by the intrinsic curiosity module. This should be scaled in order to ensure it is large enough to not be overwhelmed by extrinsic reward signals in the environment. Likewise it should not be too large to overwhelm the extrinsic reward signal. <br><br>Typical range: `0.001` - `0.1` |
| `curiosity > gamma`      | Discount factor for future rewards. <br><br>Typical range: `0.8` - `0.995` |
| `curiosity > encoding_size` | (Optional, default = `64`) Size of the encoding used by the intrinsic curiosity model. This value should be small enough to encourage the ICM to compress the original observation, but also not too small to prevent it from learning to differentiate between expected and actual observations. <br><br>Typical range: `64` - `256`  |
| `curiosity > learning_rate` | (Optional, default = `3e-4`) Learning rate used to update the intrinsic curiosity module. This should typically be decreased if training is unstable, and the curiosity loss is unstable. <br><br>Typical range: `1e-5` - `1e-3`|

**GAIL intrinsic reward**- To enable GAIL (assuming you have recorded
demonstrations), provide these settings:

| **Setting**        | **Description** |
| :------------------| :--------------- |
| `gail > strength`      | Factor by which to multiply the raw reward. Note that when using GAIL with an Extrinsic Signal, this value should be set lower if your demonstrations are suboptimal (e.g. from a human), so that a trained agent will focus on receiving extrinsic rewards instead of exactly copying the demonstrations. Keep the strength below about 0.1 in those cases. <br><br>Typical range: `0.01` - `1.0` |
| `gail > gamma`         | Discount factor for future rewards. <br><br>Typical range: `0.8` - `0.9` |
| `gail > demo_path`     | The path to your .demo file or directory of .demo files. |
| `gail > encoding_size` | (Optional, default = `64`) Size of the hidden layer used by the discriminator. This value should be small enough to encourage the discriminator to compress the original observation, but also not too small to prevent it from learning to differentiate between demonstrated and actual behavior. Dramatically increasing this size will also negatively affect training times. <br><br>Typical range: `64` - `256` |
| `gail > learning_rate` | (Optional, default = `3e-4`) Learning rate used to update the discriminator. This should typically be decreased if training is unstable, and the GAIL loss is unstable. <br><br>Typical range: `1e-5` - `1e-3` |
| `gail > use_actions`   | (Optional, default = `false`) Determines whether the discriminator should discriminate based on both observations and actions, or just observations. Set to True if you want the agent to mimic the actions from the demonstrations, and False if you'd rather have the agent visit the same states as in the demonstrations but with possibly different actions. Setting to False is more likely to be stable, especially with imperfect demonstrations, but may learn slower. |
| `gail > use_vail`      | (Optional, default = `false`) Enables a variational bottleneck within the GAIL discriminator. This forces the discriminator to learn a more general representation and reduces its tendency to be "too good" at discriminating, making learning more stable. However, it does increase training time. Enable this if you notice your imitation learning is unstable, or unable to learn the task at hand. |

**SAC-specific reward signal**- All of the reward signals configurations
described above apply to both PPO and SAC. There is one configuration for
reward signals that only applies to SAC.

| **Setting**        | **Description** |
| :------------------| :--------------- |
| `reward_signals > reward_signal_num_update`  | (Optional, default = `steps_per_update`) Number of steps per mini batch sampled and used for updating the reward signals. By default, we update the reward signals once every time the main policy is updated. However, to imitate the training procedure in certain imitation learning papers (e.g. [Kostrikov et. al](http://arxiv.org/abs/1809.02925), [Blondé et. al](http://arxiv.org/abs/1809.02064)), we may want to update the reward signal (GAIL) M times for every update of the policy. We can change `steps_per_update` of SAC to N, as well as `reward_signal_steps_per_update` under `reward_signals` to N / M to accomplish this. By default, `reward_signal_steps_per_update` is set to `steps_per_update`. |

#### Behavioral Cloning

To enable Behavioral Cloning as a pre-training option (assuming you have
recorded demonstrations), provide the following configurations under the
`behavior_cloning` section:

| **Setting**        | **Description** |
| :------------------| :--------------- |
| `demo_path`          | The path to your .demo file or directory of .demo files. |
| `strength`           | Learning rate of the imitation relative to the learning rate of PPO, and roughly corresponds to how strongly we allow BC to influence the policy. <br><br>Typical range: `0.1` - `0.5` |
| `steps`              | During BC, it is often desirable to stop using demonstrations after the agent has "seen" rewards, and allow it to optimize past the available demonstrations and/or generalize outside of the provided demonstrations. steps corresponds to the training steps over which BC is active. The learning rate of BC will anneal over the steps. Set the steps to 0 for constant imitation over the entire training run. |
| `batch_size`         | Number of demonstration experiences used for one iteration of a gradient descent update. If not specified, it will default to the `batch_size`. <br><br>Typical range: (Continuous): `512` - `5120`; (Discrete): `32` - `512` |
| `num_epoch`          | Number of passes through the experience buffer during gradient descent. If not specified, it will default to the number of epochs set for PPO. <br><br>Typical range: `3` - `10` |
| `samples_per_update` | (Optional, default = `0`) Maximum number of samples to use during each imitation update. You may want to lower this if your demonstration dataset is very large to avoid overfitting the policy on demonstrations. Set to 0 to train over all of the demonstrations at each update step. <br><br>Typical range: `buffer_size` |
| `init_path` |  Initialize trainer from a previously saved model. Note that the prior run should have used the same trainer configurations as the current run, and have been saved with the same version of ML-Agents. You should provide the full path to the folder where the checkpoints were saved, e.g. `./models/{run-id}/{behavior_name}`. This option is provided in case you want to initialize different behaviors from different runs; in most cases, it is sufficient to use the `--initialize-from` CLI parameter to initialize all models from the same run.|

#### Memory-enhanced agents using Recurrent Neural Networks

You can enable your agents to use memory, by setting `use_recurrent` to `true`
and setting `memory_size` and `sequence_length`:

| **Setting**     | **Description** |
| :-------------- | :-------------- |
| use_recurrent   | Whether to enable this option or not. |
| memory_size     | Size of the memory an agent must keep. In order to use a LSTM, training requires a sequence of experiences instead of single experiences. Corresponds to the size of the array of floating point numbers used to store the hidden state of the recurrent neural network of the policy. This value must be a multiple of 2, and should scale with the amount of information you expect the agent will need to remember in order to successfully complete the task. <br><br>Typical range: `32` - `256` |
| sequence_length | Defines how long the sequences of experiences must be while training. Note that if this number is too small, the agent will not be able to remember things over longer periods of time. If this number is too large, the neural network will take longer to train. <br><br>Typical range: `4` - `128` |

A few considerations when deciding to use memory:
- LSTM does not work well with continuous vector action space. Please use
  discrete vector action space for better results.
- Since the memories must be sent back and forth between Python and Unity,
  using too large `memory_size` will slow down training.
- Adding a recurrent layer increases the complexity of the neural network, it
  is recommended to decrease `num_layers` when using recurrent.
- It is required that `memory_size` be divisible by 4.

#### Self-Play

Training with self-play adds additional confounding factors to the usual issues
faced by reinforcement learning. In general, the tradeoff is between the skill
level and generality of the final policy and the stability of learning. Training
against a set of slowly or unchanging adversaries with low diversity results in
a more stable learning process than training against a set of quickly changing
adversaries with high diversity. With this context, this guide discusses the
exposed self-play hyperparameters and intuitions for tuning them.

If your environment contains multiple agents that are divided into teams, you
can leverage our self-play training option by providing these configurations
for each Behavior:

| **Setting**                     | **Description** |
| :------------------------------ | :-------------- |
| `save_steps` | Number of *trainer steps* between snapshots.  For example, if `save_steps=10000` then a snapshot of the current policy will be saved every `10000` trainer steps. Note, trainer steps are counted per agent. For more information, please see the [migration doc](Migrating.md) after v0.13. <br><br>A larger value of `save_steps` will yield a set of opponents that cover a wider range of skill levels and possibly play styles since the policy receives more training. As a result, the agent trains against a wider variety of opponents. Learning a policy to defeat more diverse opponents is a harder problem and so may require more overall training steps but also may lead to more general and robust policy at the end of training. This value is also dependent on how intrinsically difficult the environment is for the agent. <br><br> Typical range: `10000` - `100000` |
| `team_change` | Number of *trainer_steps* between switching the learning team. This is the number of trainer steps the teams associated with a specific ghost trainer will train before a different team becomes the new learning team. It is possible that, in asymmetric games, opposing teams require fewer trainer steps to make similar performance gains. This enables users to train a more complicated team of agents for more trainer steps than a simpler team of agents per team switch. <br><br>A larger value of `team-change` will allow the agent to train longer against it's opponents.  The longer an agent trains against the same set of opponents the more able it will be to defeat them. However, training against them for too long may result in overfitting to the particular opponent strategies and so the agent may fail against the next batch of opponents. <br><br> The value of `team-change` will determine how many snapshots of the agent's policy are saved to be used as opponents for the other team.  So, we recommend setting this value as a function of the `save_steps` parameter discussed previously. <br><br> Typical range: 4x-10x where x=`save_steps` |
| `swap_steps` | Number of *ghost steps* (not trainer steps) between swapping the opponents policy with a different snapshot. A 'ghost step' refers to a step taken by an agent *that is following a fixed policy and not learning*. The reason for this distinction is that in asymmetric games, we may have teams with an unequal number of agents e.g. a 2v1 scenario like our Strikers Vs Goalie example environment. The team with two agents collects twice as many agent steps per environment step as the team with one agent.  Thus, these two values will need to be distinct to ensure that the same number of trainer steps corresponds to the same number of opponent swaps for each team. The formula for `swap_steps` if a user desires `x` swaps of a team with `num_agents` agents against an opponent team with `num_opponent_agents` agents during `team-change` total steps is: `(num_agents / num_opponent_agents) * (team_change / x)` <br><br> Typical range: `10000` - `100000` |
| `play_against_latest_model_ratio` | Probability an agent will play against the latest opponent policy. With probability 1 - `play_against_latest_model_ratio`, the agent will play against a snapshot of its opponent from a past iteration. <br><br> A larger value of `play_against_latest_model_ratio` indicates that an agent will be playing against the current opponent more often. Since the agent is updating it's policy, the opponent will be different from iteration to iteration.  This can lead to an unstable learning environment, but poses the agent with an [auto-curricula](https://openai.com/blog/emergent-tool-use/) of more increasingly challenging situations which may lead to a stronger final policy. <br><br> Typical range: `0.0` - `1.0` |
| `window` | Size of the sliding window of past snapshots from which the agent's opponents are sampled.  For example, a `window` size of 5 will save the last 5 snapshots taken. Each time a new snapshot is taken, the oldest is discarded. A larger value of `window` means that an agent's pool of opponents will contain a larger diversity of behaviors since it will contain policies from earlier in the training run. Like in the `save_steps` hyperparameter, the agent trains against a wider variety of opponents. Learning a policy to defeat more diverse opponents is a harder problem and so may require more overall training steps but also may lead to more general and robust policy at the end of training. <br><br> Typical range: `5` - `30` |

**A Note on Reward Signals**

We make the assumption that the final reward in a trajectory corresponds to the
outcome of an episode. A final reward of +1 indicates winning, -1 indicates
losing and 0 indicates a draw. The ELO calculation (discussed below) depends on
this final reward being either +1, 0, -1.

The reward signal should still be used as described in the documentation for the
other trainers and [reward signals.](Reward-Signals.md) However, we encourage
users to be a bit more conservative when shaping reward functions due to the
instability and non-stationarity of learning in adversarial games. Specifically,
we encourage users to begin with the simplest possible reward function (+1
winning, -1 losing) and to allow for more iterations of training to compensate
for the sparsity of reward.

**Note on Swap Steps**

As an example, in a 2v1 scenario, if we want the swap to occur x=4 times during
team-change=200000 steps, the swap_steps for the team of one agent is:

swap_steps = (1 / 2) \* (200000 / 4) = 25000 The swap_steps for the team of two
agents is:

swap_steps = (2 / 1) \* (200000 / 4) = 100000 Note, with equal team sizes, the
first term is equal to 1 and swap_steps can be calculated by just dividing the
total steps by the desired number of swaps.

A larger value of swap_steps means that an agent will play against the same
fixed opponent for a longer number of training iterations. This results in a
more stable training scenario, but leaves the agent open to the risk of
overfitting it's behavior for this particular opponent. Thus, when a new
opponent is swapped, the agent may lose more often than expected.

### Curriculum Learning

To enable curriculum learning, you need to provide the `--curriculum` CLI
option and point to a YAML file that defines the curriculum. Here is one
example file:

```yml
BehaviorY:
  measure: progress
  thresholds: [0.1, 0.3, 0.5]
  min_lesson_length: 100
  signal_smoothing: true
  parameters:
    wall_height: [1.5, 2.0, 2.5, 4.0]
```

Each group of Agents under the same `Behavior Name` in an environment can have
a corresponding curriculum. These curricula are held in what we call a
"metacurriculum". A metacurriculum allows different groups of Agents to follow
different curricula within the same environment.

#### Specifying Curricula

In order to define the curricula, the first step is to decide which parameters
of the environment will vary. In the case of the Wall Jump environment, the
height of the wall is what varies. Rather than adjusting it by hand, we will
create a YAML file which describes the structure of the curricula. Within it,
we can specify which points in the training process our wall height will
change, either based on the percentage of training steps which have taken
place, or what the average reward the agent has received in the recent past is.
Below is an example config for the curricula for the Wall Jump environment.

```yaml
BigWallJump:
  measure: progress
  thresholds: [0.1, 0.3, 0.5]
  min_lesson_length: 100
  signal_smoothing: true
  parameters:
    big_wall_min_height: [0.0, 4.0, 6.0, 8.0]
    big_wall_max_height: [4.0, 7.0, 8.0, 8.0]
SmallWallJump:
  measure: progress
  thresholds: [0.1, 0.3, 0.5]
  min_lesson_length: 100
  signal_smoothing: true
  parameters:
    small_wall_height: [1.5, 2.0, 2.5, 4.0]
```

The curriculum for each Behavior has the following parameters:
| **Setting**                     | **Description** |
| :------------------------------ | :-------------- |
| `measure` |  What to measure learning progress, and advancement in lessons by.<br><br> `reward` uses a measure received reward, while `progress` uses the ratio of steps/max_steps. |
| `thresholds` | Points in value of `measure` where lesson should be increased. |
| `min_lesson_length` | The minimum number of episodes that should be completed before the lesson can change. If `measure` is set to `reward`, the average cumulative reward of the last `min_lesson_length` episodes will be used to determine if the lesson should change. Must be nonnegative. <br><br> __Important__: the average reward that is compared to the thresholds is   different than the mean reward that is logged to the console. For example, if `min_lesson_length` is `100`, the lesson will increment after the average cumulative reward of the last `100` episodes exceeds the current threshold. The mean reward logged to the console is dictated by the `summary_freq` parameter defined above. |
| `signal_smoothing` | Whether to weight the current progress measure by previous values. |
| `parameters` | Corresponds to environment parameters to control. Length of each array should be one greater than number of thresholds. |

Once our curriculum is defined, we have to use the environment parameters we
defined and modify the environment from the Agent's `OnEpisodeBegin()` function
by leveraging `Academy.Instance.EnvironmentParameters`.
See
[WallJumpAgent.cs](https://github.com/Unity-Technologies/ml-agents/blob/master/Project/Assets/ML-Agents/Examples/WallJump/Scripts/WallJumpAgent.cs)
for an example.

#### Training with a Curriculum

Once we have specified our metacurriculum and curricula, we can launch
`mlagents-learn` using the `–curriculum` flag to point to the config file for
our curricula and PPO will train using Curriculum Learning. For example, to
train agents in the Wall Jump environment with curriculum learning, we can run:

```sh
mlagents-learn config/trainer_config.yaml --curriculum=config/curricula/wall_jump.yaml --run-id=wall-jump-curriculum
```

We can then keep track of the current lessons and progresses via TensorBoard.

__Note__: If you are resuming a training session that uses curriculum, please
pass the number of the last-reached lesson using the `--lesson` flag when
running `mlagents-learn`.

### Environment Parameter Randomization

To enable parameter randomization, you need to provide the `--sampler` CLI
option and point to a YAML file that defines the curriculum. Here is one
example file:

```yaml
resampling-interval: 5000

mass:
    sampler-type: "uniform"
    min_value: 0.5
    max_value: 10

gravity:
    sampler-type: "multirange_uniform"
    intervals: [[7, 10], [15, 20]]

scale:
    sampler-type: "uniform"
    min_value: 0.75
    max_value: 3

```

Note that `mass`, `gravity` and `scale` are the names of the environment
parameters that will be sampled. If a parameter specified in the file doesn't
exist in the environment, then this parameter will be ignored.

| **Setting**                     | **Description** |
| :------------------------------ | :-------------- |
| `resampling-interval` | Number of steps for the agent to train under a particular environment configuration before resetting the environment with a new sample of `Environment Parameters`. |
| `sampler-type` | Type of sampler use for this `Environment Parameter`. This is a string that should exist in the `Sampler Factory` (explained below). |
| `sampler-type-sub-arguments` |  Specify the sub-arguments depending on the `sampler-type`. In the example above, this would correspond to the `intervals` under the `sampler-type` `multirange_uniform` for the `Environment Parameter` called `gravity`. The key name should match the name of the corresponding argument in the sampler definition (explained) below) |

Once our parameters and samplers are defined, we have to use the environment
parameters we defined and modify the environment from the Agent's
`OnEpisodeBegin()` function by leveraging
`Academy.Instance.EnvironmentParameters`.

#### Included Sampler Types

Below is a list of included `sampler-type` as part of the toolkit.

- `uniform` - Uniform sampler
    - Uniformly samples a single float value between defined endpoints. The
      sub-arguments for this sampler to specify the interval endpoints are as
      below. The sampling is done in the range of [`min_value`, `max_value`).
    - **sub-arguments** - `min_value`, `max_value`
- `gaussian` - Gaussian sampler
    - Samples a single float value from the distribution characterized by the
      mean and standard deviation. The sub-arguments to specify the Gaussian
      distribution to use are as below.
    - **sub-arguments** - `mean`, `st_dev`
- `multirange_uniform` - Multirange uniform sampler
    - Uniformly samples a single float value between the specified intervals.
      Samples by first performing a weight pick of an interval from the list
      of intervals (weighted based on interval width) and samples uniformly
      from the selected interval (half-closed interval, same as the uniform
      sampler). This sampler can take an arbitrary number of intervals in a
      list in the following format:
      [[`interval_1_min`, `interval_1_max`], [`interval_2_min`, `interval_2_max`], ...]
    - **sub-arguments** - `intervals`

The implementation of the samplers can be found at
`ml-agents-envs/mlagents_envs/sampler_class.py`.

#### Defining a New Sampler Type

If you want to define your own sampler type, you must first inherit the
*Sampler* base class (included in the `sampler_class` file) and preserve the
interface. Once the class for the required method is specified, it must be
registered in the Sampler Factory.

This can be done by subscribing to the *register_sampler* method of the
`SamplerFactory`. The command is as follows:

```SamplerFactory.register_sampler(*custom_sampler_string_key*, *custom_sampler_object*)```

Once the Sampler Factory reflects the new register, the new sampler type can be
used for sample any `Environment Parameter`. For example, lets say a new
sampler type was implemented as below and we register the `CustomSampler` class
with the string `custom-sampler` in the Sampler Factory.

```python
class CustomSampler(Sampler):

    def __init__(self, argA, argB, argC):
        self.possible_vals = [argA, argB, argC]

    def sample_all(self):
        return np.random.choice(self.possible_vals)
```

Now we need to specify the new sampler type in the sampler YAML file. For
example, we use this new sampler type for the `Environment Parameter` *mass*.

```yaml
mass:
    sampler-type: "custom-sampler"
    argB: 1
    argA: 2
    argC: 3
```

#### Training with Environment Parameter Randomization

After the sampler YAML file is defined, we proceed by launching
`mlagents-learn` and specify our configured sampler file with the `--sampler`
flag. For example, if we wanted to train the 3D ball agent with parameter
randomization using `Environment Parameters` with
`config/3dball_randomize.yaml` sampling setup, we would run

```sh
mlagents-learn config/trainer_config.yaml --sampler=config/3dball_randomize.yaml
--run-id=3D-Ball-randomize
```

We can observe progress and metrics via Tensorboard.

### Training Using Concurrent Unity Instances

In order to run concurrent Unity instances during training, set the number of
environment instances using the command line option `--num-envs=<n>` when you
invoke `mlagents-learn`. Optionally, you can also set the `--base-port`, which
is the starting port used for the concurrent Unity instances.

Some considerations:
- **Buffer Size** - If you are having trouble getting an agent to train, even
  with multiple concurrent Unity instances, you could increase `buffer_size` in
  the `config/trainer_config.yaml` file. A common practice is to multiply
  `buffer_size` by `num-envs`.
- **Resource Constraints** - Invoking concurrent Unity instances is constrained
  by the resources on the machine. Please use discretion when setting
  `--num-envs=<n>`.
- **Result Variation Using Concurrent Unity Instances** - If you keep all the
  hyperparameters the same, but change `--num-envs=<n>`, the results and model
  would likely change.
