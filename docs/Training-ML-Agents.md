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

| **Setting**            | **Range** | **Description** |
| :--------------------- | :-------------- | :-------------- |
| `trainer`                | `ppo` or `sac` | The type of training to perform. |
| `init_path`              | relative path | Initialize trainer from a previously saved model. Note that the prior run should have used the same trainer configurations as the current run, and have been saved with the same version of ML-Agents. You should provide the full path to the folder where the checkpoints were saved, e.g. `./models/{run-id}/{behavior_name}`. This option is provided in case you want to initialize different behaviors from different runs; in most cases, it is sufficient to use the `--initialize-from` CLI parameter to initialize all models from the same run.|
| `summary_freq`           | | The number of experiences that needs to be collected before generating and displaying training statistics. This determines the granularity of the graphs in Tensorboard.             |
| `batch_size`             | (Continuous - PPO): `512` - `5120`; (Continuous - SAC): `128` - `1024`; (Discrete, PPO & SAC): `32` - `512` | The number of experiences in each iteration of gradient descent. **This should always be a fraction of the `buffer_size`**. If you are using a continuous action space, this value should be large (in the order of 1000s). If you are using a discrete action space, this value should be smaller (in order of 10s). |
| `buffer_size`            | PPO: `2048` - `409600`; SAC: `50000` - `1000000` | The number of experiences to collect before updating the policy model. Corresponds to how many experiences (agent observations, actions and rewards obtained) should be collected before we do any learning or updating of the model. **This should be a multiple of `batch_size`**. Typically a larger `buffer_size` corresponds to more stable training updates. In SAC, the max size of the experience buffer - on the order of thousands of times longer than your episodes, so that SAC can learn from old as well as new experiences. |
| `hidden_units`           | `32` - `512` | The number of units in the hidden layers of the neural network. Correspond to how many units are in each fully connected layer of the neural network. For simple problems where the correct action is a straightforward combination of the observation inputs, this should be small. For problems where the action is a very complex interaction between the observation variables, this should be larger.|
| `learning_rate`          | `1e-5` - `1e-3` | The initial learning rate for gradient descent. Corresponds to the strength of each gradient descent update step. This should typically be decreased if training is unstable, and the reward does not consistently increase.|
| `learning_rate_schedule` | `linear` (default) or `constant` | Determines how learning rate changes over time. For PPO, we recommend decaying learning rate until max_steps so learning converges more stably. However, for some cases (e.g. training for an unknown amount of time) this feature can be disabled. For SAC, we recommend holding learning rate constant so that the agent can continue to learn until its Q function converges naturally. `linear` (default) decays the learning_rate linearly, reaching 0 at max_steps, while `constant` keeps the learning rate constant for the entire training run. |
| `max_steps`              | `5e5` - `1e7` | Corresponds to the total number of experience points that must be collected from the simulation before ending the training process. |
| `normalize`              | `true` or `false` | Corresponds to whether normalization is applied to the vector observation inputs. This normalization is based on the running average and variance of the vector observation. Normalization can be helpful in cases with complex continuous control problems, but may be harmful with simpler discrete control problems. |
| `num_layers`             | `1` - `3` | The number of hidden layers in the neural network. Corresponds to how many hidden layers are present after the observation input, or after the CNN encoding of the visual observation. For simple problems, fewer layers are likely to train faster and more efficiently. More layers may be necessary for more complex control problems. |
| `time_horizon`           | `32` - `2048` | Corresponds to how many steps of experience to collect per-agent before adding it to the experience buffer. When this limit is reached before the end of an episode, a value estimate is used to predict the overall expected reward from the agent's current state. As such, this parameter trades off between a less biased, but higher variance estimate (long time horizon) and more biased, but less varied estimate (short time horizon). In cases where there are frequent rewards within an episode, or episodes are prohibitively large, a smaller number can be more ideal. This number should be large enough to capture all the important behavior within a sequence of an agent's actions. |
| `vis_encoder_type`       | `simple` (default), `nature_cnn` or `resnet` | Corresponds to the encoder type for encoding visual observations. `simple` (default) uses a simple encoder which consists of two convolutional layers, `nature_cnn` uses the CNN implementation proposed by [Mnih et al.](https://www.nature.com/articles/nature14236), consisting of three convolutional layers, and `resnet` uses the [IMPALA Resnet](https://arxiv.org/abs/1802.01561) consisting of three stacked layers, each with two residual blocks, making a much larger network than the other two. |

#### Trainer-specific Configurations

Depending on your choice of a trainer, there are additional trainer-specific
configurations. We present them below in two separate tables, but keep in mind
that you only need to include the configurations for the trainer selected (i.e.
the `trainer` setting above).

PPO-specific configurations:

| **Setting** | **Range** | **Description** |
| :---------- | :-------- |:--------------- |
| `beta`        | `1e-4` - `1e-2` | Corresponds to the strength of the entropy regularization, which makes the policy "more random." This ensures that agents properly explore the action space during training. Increasing this will ensure more random actions are taken. This should be adjusted such that the entropy (measurable from TensorBoard) slowly decreases alongside increases in reward. If entropy drops too quickly, increase beta. If entropy drops too slowly, decrease `beta`. |
| `epsilon`     | `0.1` - `0.3` | Influences how rapidly the policy can evolve during training. Corresponds to the acceptable threshold of divergence between the old and new policies during gradient descent updating. Setting this value small will result in more stable updates, but will also slow the training process. |
| `lambd`       | `0.9` - `0.95` | The regularization parameter. Corresponds to the lambda parameter used when calculating the Generalized Advantage Estimate ([GAE](https://arxiv.org/abs/1506.02438)). This can be thought of as how much the agent relies on its current value estimate when calculating an updated value estimate. Low values correspond to relying more on the current value estimate (which can be high bias), and high values correspond to relying more on the actual rewards received in the environment (which can be high variance). The parameter provides a trade-off between the two, and the right value can lead to a more stable training process. |
| `num_epoch`   | `3` - `10` | The number of passes to make through the experience buffer when performing gradient descent optimization.The larger the batch_size, the larger it is acceptable to make this. Decreasing this will ensure more stable updates, at the cost of slower learning. |

SAC-specific configurations:

| **Setting**        | **Range** | **Description** |
| :------------------| :-------- |:--------------- |
| `buffer_init_steps`  | `1000` - `10000` | The number of experiences to collect into the buffer before updating the policy model. As the untrained policy is fairly random, pre-filling the buffer with random actions is useful for exploration. Typically, at least several episodes of experiences should be pre-filled. |
| `init_entcoef`       | (Continuous): `0.5` - `1.0`; (Discrete): `0.05` - `0.5` | How much the agent should explore in the beginning of training. Corresponds to the initial entropy coefficient set at the beginning of training. In SAC, the agent is incentivized to make its actions entropic to facilitate better exploration. The entropy coefficient weighs the true reward with a bonus entropy reward. The entropy coefficient is [automatically adjusted](https://arxiv.org/abs/1812.05905) to a preset target entropy, so the `init_entcoef` only corresponds to the starting value of the entropy bonus. Increase init_entcoef to explore more in the beginning, decrease to converge to a solution faster. |
| `save_replay_buffer` | `true` or `false` (default) | Enables you to save and load the experience replay buffer as well as the model when quitting and re-starting training. This may help resumes go more smoothly, as the experiences collected won't be wiped. Note that replay buffers can be very large, and will take up a considerable amount of disk space. For that reason, we disable this feature by default. |
| `tau`                | `0.005` - `0.01` | How aggressively to update the target network used for bootstrapping value estimation in SAC. Corresponds to the magnitude of the target Q update during the SAC model update. In SAC, there are two neural networks: the target and the policy. The target network is used to bootstrap the policy's estimate of the future rewards at a given state, and is fixed while the policy is being updated. This target is then slowly updated according to tau. Typically, this value should be left at 0.005. For simple problems, increasing tau to 0.01 might reduce the time it takes to learn, at the cost of stability.
| `steps_per_update`     | `1` - `20` | Corresponds to the average ratio of agent steps (actions) taken to updates made of the agent's policy. In SAC, a single "update" corresponds to grabbing a batch of size `batch_size` from the experience replay buffer, and using this mini batch to update the models. Note that it is not guaranteed that after exactly `steps_per_update` steps an update will be made, only that the ratio will hold true over many steps. Typically, `steps_per_update` should be greater than or equal to 1. Note that setting `steps_per_update` lower will improve sample efficiency (reduce the number of steps required to train) but increase the CPU time spent performing updates. For most environments where steps are fairly fast (e.g. our example environments) `steps_per_update` equal to the number of agents in the scene is a good balance. For slow environments (steps take 0.1 seconds or more) reducing `steps_per_update` may improve training speed. We can also change `steps_per_update` to lower than 1 to update more often than once per step, though this will usually result in a slowdown unless the environment is very slow.
| `train_interval` | `1` - `5` | Corresponds to the number of steps taken between each agent training event. Typically, we can train after every step, but if your environment's steps are very small and very frequent, there may not be any new interesting information between steps, and `train_interval` can be increased. |

#### Memory-enhanced agents using Recurrent Neural Networks

#### Behavioral Cloning

#### Reward Signals

#### Self-Play


### Curriculum Learning


### Environment Parameter Randomization


### Training Using Concurrent Unity Instances
