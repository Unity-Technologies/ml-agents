# Training ML-Agents

**Table of Contents**

- [Training with mlagents-learn](#training-with-mlagents-learn)
  - [Starting Training](#starting-training)
    - [Observing Training](#observing-training)
    - [Stopping and Resuming Training](#stopping-and-resuming-training)
    - [Loading an Existing Model](#loading-an-existing-model)
- [Training Configurations](#training-configurations)
  - [Behavior Configurations](#behavior-configurations)
  - [Curriculum Learning](#curriculum-learning)
    - [Specifying Curricula](#specifying-curricula)
    - [Training with a Curriculum](#training-with-a-curriculum)
  - [Environment Parameter Randomization](#environment-parameter-randomization)
    - [Included Sampler Types](#included-sampler-types)
    - [Defining a New Sampler Type](#defining-a-new-sampler-type)
    - [Training with Environment Parameter Randomization](#training-with-environment-parameter-randomization)
  - [Training Using Concurrent Unity Instances](#training-using-concurrent-unity-instances)

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
  structure of this file and the meaning of the hyperparameters (and advice on
  how to set them) in the dedicated
  [Training Configurations](#training-configurations) section below.
- `<env_name>`**(Optional)** is the name (including path) of your
  [Unity executable](Learning-Environment-Executable.md) containing the agents
  to be trained. If `<env_name>` is not passed, the training will happen in the
  Editor. Press the **Play** button in Unity when the message _"Start training
  by pressing the Play button in the Unity Editor"_ is displayed on the screen.
- `<run-identifier>` is a unique name you can use to identify the results of
  your training runs.

See the
[Getting Started Guide](Getting-Started.md#training-a-new-model-with-reinforcement-learning)
for a sample execution of the `mlagents-learn` command.

#### Observing Training

Regardless of which training methods, configurations or hyperparameters you
provide, the training process will always generate three artifacts, all found
in the `results/<run-identifier>` folder:

1. Summaries: these are training metrics that
   are updated throughout the training process. They are helpful to monitor your
   training performance and may help inform how to update your hyperparameter
   values. See [Using TensorBoard](Using-Tensorboard.md) for more details on how
   to visualize the training metrics.
1. Models: these contain the model checkpoints that
   are updated throughout training and the final model file (`.nn`). This final
   model file is generated once either when training completes or is
   interrupted.
1. Timers file (under `results/<run-identifier>/run_logs`): this contains aggregated
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

More specifically, this section offers a detailed guide on the command-line
flags for `mlagents-learn` that control the training configurations:

- `<trainer-config-file>`: defines the training hyperparameters for each
  Behavior in the scene, and the set-ups for Curriculum Learning and
  Environment Parameter Randomization
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
`config/ppo/3DBall.yaml` was used to train the 3D Balance Ball in the
[Getting Started](Getting-Started.md) guide. That configuration file uses the
PPO trainer, but we also have configuration files for SAC and GAIL.

Additionally, the set of configurations you provide depend on the training
functionalities you use (see [ML-Agents Toolkit Overview](ML-Agents-Overview.md)
for a description of all the training functionalities). Each functionality you
add typically has its own training configurations. For instance:

- Use PPO or SAC?
- Use Recurrent Neural Networks for adding memory to your agents?
- Use the intrinsic curiosity module?
- Ignore the environment reward signal?
- Pre-train using behavioral cloning? (Assuming you have recorded
  demonstrations.)
- Include the GAIL intrinsic reward signals? (Assuming you have recorded
  demonstrations.)
- Use self-play? (Assuming your environment includes multiple agents.)


The trainer config file, `<trainer-config-file>`, determines the features you will
use during training, and the answers to the above questions will dictate its contents.
The rest of this guide breaks down the different sub-sections of the trainer config file
and explains the possible settings for each.

**NOTE:** The configuration file format has been changed from 0.17.0 and onwards. To convert
an old set of configuration files (trainer config, curriculum, and sampler files) to the new
format, a script has been provided. Run `python config/upgrade_config.py -h` in your  console
to see the script's usage.

### Behavior Configurations

The primary section of the trainer config file is a
set of configurations for each Behavior in your scene. These are defined under
the sub-section `behaviors` in your trainer config file. Some of the
configurations are required while others are optional. To help us get started,
below is a sample file that includes all the possible settings if we're using a
PPO trainer with all the possible training functionalities enabled (memory,
behavioral cloning, curiosity, GAIL and self-play). You will notice that
curriculum and environment parameter randomization settings are not part of the `behaviors`
configuration, but their settings live in different sections that we'll cover subsequently.

```yaml
behaviors:
  BehaviorPPO:
    trainer_type: ppo

    hyperparameters:
      # Hyperparameters common to PPO and SAC
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear

      # PPO-specific hyperparameters
      # Replaces the "PPO-specific hyperparameters" section above
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3

    # Configuration of the neural network (common to PPO/SAC)
    network_settings:
      vis_encoder_type: simple
      normalize: false
      hidden_units: 128
      num_layers: 2
      # memory
      memory:
        sequence_length: 64
        memory_size: 256

    # Trainer configurations common to all trainers
    max_steps: 5.0e5
    time_horizon: 64
    summary_freq: 10000
    keep_checkpoints: 5
    threaded: true
    init_path: null

    # behavior cloning
    behavioral_cloning:
      demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
      strength: 0.5
      steps: 150000
      batch_size: 512
      num_epoch: 3
      samples_per_update: 0

    reward_signals:
      # environment reward (default)
      extrinsic:
        strength: 1.0
        gamma: 0.99

      # curiosity module
      curiosity:
        strength: 0.02
        gamma: 0.99
        encoding_size: 256
        learning_rate: 3.0e-4

      # GAIL
      gail:
        strength: 0.01
        gamma: 0.99
        encoding_size: 128
        demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
        learning_rate: 3.0e-4
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
behaviors:
  BehaviorSAC:
    trainer_type: sac

    # Trainer configs common to PPO/SAC (excluding reward signals)
    # same as PPO config

    # SAC-specific configs (replaces the hyperparameters section above)
    hyperparameters:
      # Hyperparameters common to PPO and SAC
      # Same as PPO config

      # SAC-specific hyperparameters
      # Replaces the "PPO-specific hyperparameters" section above
      buffer_init_steps: 0
      tau: 0.005
      steps_per_update: 10.0
      save_replay_buffer: false
      init_entcoef: 0.5
      reward_signal_steps_per_update: 10.0

    # Configuration of the neural network (common to PPO/SAC)
    network_settings:
      # Same as PPO config

    # Trainer configurations common to all trainers
      # <Same as PPO config>

    # pre-training using behavior cloning
    behavioral_cloning:
      # same as PPO config

    reward_signals:
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
each of these parameters mean and provide guidelines on how to set them. See
[Training Configuration File](Training-Configuration-File.md) for a detailed
description of all the configurations listed above, along with their defaults.
Unless otherwise specified, omitting a configuration will revert it to its default.

### Curriculum Learning

To enable curriculum learning, you need to add a `curriculum ` sub-section to the trainer
configuration YAML file. Within this sub-section, add an entry for each behavior that defines
the curriculum for thatbehavior. Here is one example:

```yml
behaviors:
  BehaviorY:
    # < Same as above >

# Add this section
curriculum:
  BehaviorY:
    measure: progress
    thresholds: [0.1, 0.3, 0.5]
    min_lesson_length: 100
    signal_smoothing: true
    parameters:
      wall_height: [1.5, 2.0, 2.5, 4.0]
```

Each group of Agents under the same `Behavior Name` in an environment can have a
corresponding curriculum. These curricula are held in what we call a
"metacurriculum". A metacurriculum allows different groups of Agents to follow
different curricula within the same environment.

#### Specifying Curricula

In order to define the curricula, the first step is to decide which parameters
of the environment will vary. In the case of the Wall Jump environment, the
height of the wall is what varies. Rather than adjusting it by hand, we will
create a configuration which describes the structure of the curricula. Within it, we
can specify which points in the training process our wall height will change,
either based on the percentage of training steps which have taken place, or what
the average reward the agent has received in the recent past is. Below is an
example config for the curricula for the Wall Jump environment.

```yaml
behaviors:
  BigWallJump:
    # < Trainer parameters for BigWallJump >
  SmallWallJump:
    # < Trainer parameters for SmallWallJump >

curriculum:
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

| **Setting**         | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `measure`           | What to measure learning progress, and advancement in lessons by.<br><br> `reward` uses a measure received reward, while `progress` uses the ratio of steps/max_steps.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `thresholds`        | Points in value of `measure` where lesson should be increased.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `min_lesson_length` | The minimum number of episodes that should be completed before the lesson can change. If `measure` is set to `reward`, the average cumulative reward of the last `min_lesson_length` episodes will be used to determine if the lesson should change. Must be nonnegative. <br><br> **Important**: the average reward that is compared to the thresholds is different than the mean reward that is logged to the console. For example, if `min_lesson_length` is `100`, the lesson will increment after the average cumulative reward of the last `100` episodes exceeds the current threshold. The mean reward logged to the console is dictated by the `summary_freq` parameter defined above. |
| `signal_smoothing`  | Whether to weight the current progress measure by previous values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `parameters`        | Corresponds to environment parameters to control. Length of each array should be one greater than number of thresholds.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

#### Training with a Curriculum

Once we have specified our metacurriculum and curricula, we can launch
`mlagents-learn` to point to the config file containing
our curricula and PPO will train using Curriculum Learning. For example, to
train agents in the Wall Jump environment with curriculum learning, we can run:

```sh
mlagents-learn config/ppo/WallJump_curriculum.yaml --run-id=wall-jump-curriculum
```

We can then keep track of the current lessons and progresses via TensorBoard.

**Note**: If you are resuming a training session that uses curriculum, please
pass the number of the last-reached lesson using the `--lesson` flag when
running `mlagents-learn`.

### Environment Parameter Randomization

To enable parameter randomization, you need to add a `parameter-randomization` sub-section
to your trainer config YAML file. Here is one example:

```yaml
behaviors:
  # < Same as above>

parameter_randomization:
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

| **Setting**                  | **Description**                                                                                                                                                                                                                                                                                                                         |
| :--------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resampling-interval`        | Number of steps for the agent to train under a particular environment configuration before resetting the environment with a new sample of `Environment Parameters`.                                                                                                                                                                     |
| `sampler-type`               | Type of sampler use for this `Environment Parameter`. This is a string that should exist in the `Sampler Factory` (explained below).                                                                                                                                                                                                    |
| `sampler-type-sub-arguments` | Specify the sub-arguments depending on the `sampler-type`. In the example above, this would correspond to the `intervals` under the `sampler-type` `multirange_uniform` for the `Environment Parameter` called `gravity`. The key name should match the name of the corresponding argument in the sampler definition (explained) below) |

#### Included Sampler Types

Below is a list of included `sampler-type` as part of the toolkit.

- `uniform` - Uniform sampler
  - Uniformly samples a single float value between defined endpoints. The
    sub-arguments for this sampler to specify the interval endpoints are as
    below. The sampling is done in the range of [`min_value`, `max_value`).
  - **sub-arguments** - `min_value`, `max_value`
- `gaussian` - Gaussian sampler
  - Samples a single float value from the distribution characterized by the mean
    and standard deviation. The sub-arguments to specify the Gaussian
    distribution to use are as below.
  - **sub-arguments** - `mean`, `st_dev`
- `multirange_uniform` - Multirange uniform sampler
  - Uniformly samples a single float value between the specified intervals.
    Samples by first performing a weight pick of an interval from the list of
    intervals (weighted based on interval width) and samples uniformly from the
    selected interval (half-closed interval, same as the uniform sampler). This
    sampler can take an arbitrary number of intervals in a list in the following
    format: [[`interval_1_min`, `interval_1_max`], [`interval_2_min`,
    `interval_2_max`], ...]
  - **sub-arguments** - `intervals`

The implementation of the samplers can be found in the
[sampler_class.py file](../ml-agents/mlagents/trainers/sampler_class.py).

#### Defining a New Sampler Type

If you want to define your own sampler type, you must first inherit the
_Sampler_ base class (included in the `sampler_class` file) and preserve the
interface. Once the class for the required method is specified, it must be
registered in the Sampler Factory.

This can be done by subscribing to the _register_sampler_ method of the
`SamplerFactory`. The command is as follows:

`SamplerFactory.register_sampler(*custom_sampler_string_key*, *custom_sampler_object*)`

Once the Sampler Factory reflects the new register, the new sampler type can be
used for sample any `Environment Parameter`. For example, lets say a new sampler
type was implemented as below and we register the `CustomSampler` class with the
string `custom-sampler` in the Sampler Factory.

```python
class CustomSampler(Sampler):

    def __init__(self, argA, argB, argC):
        self.possible_vals = [argA, argB, argC]

    def sample_all(self):
        return np.random.choice(self.possible_vals)
```

Now we need to specify the new sampler type in the sampler YAML file. For
example, we use this new sampler type for the `Environment Parameter` _mass_.

```yaml
mass:
  sampler-type: "custom-sampler"
  argB: 1
  argA: 2
  argC: 3
```

#### Training with Environment Parameter Randomization

After the sampler configuration is defined, we proceed by launching `mlagents-learn`
and specify trainer configuration with `parameter-randomization` defined. For example,
if we wanted to train the 3D ball agent with parameter randomization using
`Environment Parameters` with sampling setup, we would run

```sh
mlagents-learn config/ppo/3DBall_randomize.yaml --run-id=3D-Ball-randomize
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
  the trainer config file. A common practice is to multiply
  `buffer_size` by `num-envs`.
- **Resource Constraints** - Invoking concurrent Unity instances is constrained
  by the resources on the machine. Please use discretion when setting
  `--num-envs=<n>`.
- **Result Variation Using Concurrent Unity Instances** - If you keep all the
  hyperparameters the same, but change `--num-envs=<n>`, the results and model
  would likely change.
