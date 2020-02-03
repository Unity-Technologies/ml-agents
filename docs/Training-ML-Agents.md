# Training ML-Agents

The ML-Agents toolkit conducts training using an external Python training
process. During training, this external process communicates with the Academy
to generate a block of agent experiences. These
experiences become the training set for a neural network used to optimize the
agent's policy (which is essentially a mathematical function mapping
observations to actions). In reinforcement learning, the neural network
optimizes the policy by maximizing the expected rewards. In imitation learning,
the neural network optimizes the policy to achieve the smallest difference
between the actions chosen by the agent trainee and the actions chosen by the
expert in the same situation.

The output of the training process is a model file containing the optimized
policy. This model file is a TensorFlow data graph containing the mathematical
operations and the optimized weights selected during the training process. You
can set the generated model file in the Behaviors Parameters under your
Agent in your Unity project to decide the best course of action for an agent.

Use the command `mlagents-learn` to train your agents. This command is installed
with the `mlagents` package and its implementation can be found at
`ml-agents/mlagents/trainers/learn.py`. The [configuration file](#training-config-file),
like `config/trainer_config.yaml` specifies the hyperparameters used during training.
You can edit this file with a text editor to add a specific configuration for
each Behavior.

For a broader overview of reinforcement learning, imitation learning and the
ML-Agents training process, see [ML-Agents Toolkit
Overview](ML-Agents-Overview.md).

## Training with mlagents-learn

Use the `mlagents-learn` command to train agents. `mlagents-learn` supports
training with
[reinforcement learning](Background-Machine-Learning.md#reinforcement-learning),
[curriculum learning](Training-Curriculum-Learning.md),
and [behavioral cloning imitation learning](Training-Imitation-Learning.md).

Run `mlagents-learn` from the command line to launch the training process. Use
the command line patterns and the `config/trainer_config.yaml` file to control
training options.

The basic command for training is:

```sh
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier> --train
```

where

* `<trainer-config-file>` is the file path of the trainer configuration yaml.
* `<env_name>`__(Optional)__ is the name (including path) of your Unity
  executable containing the agents to be trained. If `<env_name>` is not passed,
  the training will happen in the Editor. Press the :arrow_forward: button in
  Unity when the message _"Start training by pressing the Play button in the
  Unity Editor"_ is displayed on the screen.
* `<run-identifier>` is an optional identifier you can use to identify the
  results of individual training runs.

For example, suppose you have a project in Unity named "CatsOnBicycles" which
contains agents ready to train. To perform the training:

1. [Build the project](Learning-Environment-Executable.md), making sure that you
   only include the training scene.
2. Open a terminal or console window.
3. Navigate to the directory where you installed the ML-Agents Toolkit.
4. Run the following to launch the training process using the path to the Unity
   environment you built in step 1:

```sh
mlagents-learn config/trainer_config.yaml --env=../../projects/Cats/CatsOnBicycles.app --run-id=cob_1 --train
```

During a training session, the training program prints out and saves updates at
regular intervals (specified by the `summary_freq` option). The saved statistics
are grouped by the `run-id` value so you should assign a unique id to each
training run if you plan to view the statistics. You can view these statistics
using TensorBoard during or after training by running the following command:

```sh
tensorboard --logdir=summaries --port 6006
```

And then opening the URL: [localhost:6006](http://localhost:6006).

**Note:** The default port TensorBoard uses is 6006. If there is an existing session
running on port 6006 a new session can be launched on an open port using the --port
option.

When training is finished, you can find the saved model in the `models` folder
under the assigned run-id — in the cats example, the path to the model would be
`models/cob_1/CatsOnBicycles_cob_1.nn`.

While this example used the default training hyperparameters, you can edit the
[training_config.yaml file](#training-config-file) with a text editor to set
different values.

### Command Line Training Options

In addition to passing the path of the Unity executable containing your training
environment, you can set the following command line options when invoking
`mlagents-learn`:

* `--env=<env>`: Specify an executable environment to train.
* `--curriculum=<file>`: Specify a curriculum JSON file for defining the
  lessons for curriculum training. See [Curriculum
  Training](Training-Curriculum-Learning.md) for more information.
* `--sampler=<file>`: Specify a sampler YAML file for defining the
  sampler for generalization training. See [Generalization
  Training](Training-Generalized-Reinforcement-Learning-Agents.md) for more information.
* `--keep-checkpoints=<n>`: Specify the maximum number of model checkpoints to
  keep. Checkpoints are saved after the number of steps specified by the
  `save-freq` option. Once the maximum number of checkpoints has been reached,
  the oldest checkpoint is deleted when saving a new checkpoint. Defaults to 5.
* `--lesson=<n>`: Specify which lesson to start with when performing curriculum
  training. Defaults to 0.
* `--num-envs=<n>`: Specifies the number of concurrent Unity environment instances to
  collect experiences from when training. Defaults to 1.
* `--run-id=<path>`: Specifies an identifier for each training run. This
  identifier is used to name the subdirectories in which the trained model and
  summary statistics are saved as well as the saved model itself. The default id
  is "ppo". If you use TensorBoard to view the training statistics, always set a
  unique run-id for each training run. (The statistics for all runs with the
  same id are combined as if they were produced by a the same session.)
* `--save-freq=<n>`: Specifies how often (in  steps) to save the model during
  training. Defaults to 50000.
* `--seed=<n>`: Specifies a number to use as a seed for the random number
  generator used by the training code.
* `--env-args=<string>`: Specify arguments for the executable environment. Be aware that
  the standalone build will also process these as
  [Unity Command Line Arguments](https://docs.unity3d.com/Manual/CommandLineArguments.html).
  You should choose different argument names if you want to create environment-specific arguments.
  All arguments after this flag will be passed to the executable. For example, setting
  `mlagents-learn config/trainer_config.yaml --env-args --num-orcs 42` would result in
   ` --num-orcs 42` passed to the executable.
* `--base-port`: Specifies the starting port. Each concurrent Unity environment instance
  will get assigned a port sequentially, starting from the `base-port`. Each instance
  will use the port `(base_port + worker_id)`, where the `worker_id` is sequential IDs
  given to each instance from 0 to `num_envs - 1`. Default is 5005. __Note:__ When
  training using the Editor rather than an executable, the base port will be ignored.
* `--train`: Specifies whether to train model or only run in inference mode.
  When training, **always** use the `--train` option.
* `--load`: If set, the training code loads an already trained model to
  initialize the neural network before training. The learning code looks for the
  model in `models/<run-id>/` (which is also where it saves models at the end of
  training). When not set (the default), the neural network weights are randomly
  initialized and an existing model is not loaded.
* `--no-graphics`: Specify this option to run the Unity executable in
  `-batchmode` and doesn't initialize the graphics driver. Use this only if your
  training doesn't involve visual observations (reading from Pixels). See
  [here](https://docs.unity3d.com/Manual/CommandLineArguments.html) for more
  details.
* `--debug`: Specify this option to enable debug-level logging for some parts of the code.
* `--multi-gpu`: Setting this flag enables the use of multiple GPU's (if available) during training.
* `--cpu`: Forces training using CPU only.
* Engine Configuration :
  * `--width' : The width of the executable window of the environment(s) in pixels
  (ignored for editor training) (Default 84)
  * `--height` : The height of the executable window of the environment(s) in pixels
  (ignored for editor training). (Default 84)
  * `--quality-level` : The quality level of the environment(s). Equivalent to
  calling `QualitySettings.SetQualityLevel` in Unity. (Default 5)
  * `--time-scale` : The time scale of the Unity environment(s). Equivalent to setting
  `Time.timeScale` in Unity. (Default 20.0, maximum 100.0)
  * `--target-frame-rate` : The target frame rate of the Unity environment(s).
  Equivalent to setting `Application.targetFrameRate` in Unity. (Default: -1)

### Training Config File

The training config files `config/trainer_config.yaml`, `config/sac_trainer_config.yaml`,
`config/gail_config.yaml` and `config/offline_bc_config.yaml` specifies the training method,
the hyperparameters, and a few additional values to use when training with Proximal Policy
Optimization(PPO), Soft Actor-Critic(SAC), GAIL (Generative Adversarial Imitation Learning)
with PPO/SAC, and Behavioral Cloning(BC)/Imitation with PPO/SAC. These files are divided
into sections. The **default** section defines the default values for all the available
training with PPO, SAC, GAIL (with PPO), and BC. These files are divided into sections.
The **default** section defines the default values for all the available settings. You can
also add new sections to override these defaults to train specific Behaviors. Name each of these
override sections after the appropriate `Behavior Name`. Sections for the
example environments are included in the provided config file.

|     **Setting**      |                                                                                     **Description**                                                                                     | **Applies To Trainer\*** |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------- |
| batch_size           | The number of experiences in each iteration of gradient descent.                                                                                                                        | PPO, SAC             |
| batches_per_epoch    | In imitation learning, the number of batches of training examples to collect before training the model.                                                                                 |                        |
| beta                 | The strength of entropy regularization.                                                                                                                                                 | PPO                      |
| buffer_size          | The number of experiences to collect before updating the policy model. In SAC, the max size of the experience buffer.                                                                   | PPO, SAC                 |
| buffer_init_steps    | The number of experiences to collect into the buffer before updating the policy model.                                                                                                  | SAC                      |
| epsilon              | Influences how rapidly the policy can evolve during training.                                                                                                                           | PPO                      |
| hidden_units         | The number of units in the hidden layers of the neural network.                                                                                                                         | PPO, SAC             |
| init_entcoef         | How much the agent should explore in the beginning of training.                                                                                                                         | SAC                      |
| lambd                | The regularization parameter.                                                                                                                                                           | PPO                      |
| learning_rate        | The initial learning rate for gradient descent.                                                                                                                                         | PPO, SAC             |
| max_steps            | The maximum number of simulation steps to run during a training session.                                                                                                                | PPO, SAC             |
| memory_size          | The size of the memory an agent must keep. Used for training with a recurrent neural network. See [Using Recurrent Neural Networks](Feature-Memory.md).                                 | PPO, SAC             |
| normalize            | Whether to automatically normalize observations.                                                                                                                                        | PPO, SAC                 |
| num_epoch            | The number of passes to make through the experience buffer when performing gradient descent optimization.                                                                               | PPO                      |
| num_layers           | The number of hidden layers in the neural network.                                                                                                                                      | PPO, SAC             |
| behavioral_cloning          | Use demonstrations to bootstrap the policy neural network. See [Pretraining Using Demonstrations](Training-PPO.md#optional-behavioral-cloning-using-demonstrations).                           | PPO, SAC                 |
| reward_signals       | The reward signals used to train the policy. Enable Curiosity and GAIL here. See [Reward Signals](Reward-Signals.md) for configuration options.                                         | PPO, SAC             |
| save_replay_buffer   | Saves the replay buffer when exiting training, and loads it on resume.                                                                                                                  | SAC                      |
| sequence_length      | Defines how long the sequences of experiences must be while training. Only used for training with a recurrent neural network. See [Using Recurrent Neural Networks](Feature-Memory.md). | PPO, SAC             |
| summary_freq         | How often, in steps, to save training statistics. This determines the number of data points shown by TensorBoard.                                                                       | PPO, SAC             |
| tau                  | How aggressively to update the target network used for bootstrapping value estimation in SAC.                                                                                           | SAC                      |
| time_horizon         | How many steps of experience to collect per-agent before adding it to the experience buffer.                                                                                            | PPO, SAC    |
| trainer              | The type of training to perform: "ppo", "sac", "offline_bc" or "online_bc".                                                                                                             | PPO, SAC             |
| train_interval       | How often to update the agent.                                                                                                                                                          | SAC                      |
| num_update           | Number of mini-batches to update the agent with during each update.                                                                                                                     | SAC                      |
| use_recurrent        | Train using a recurrent neural network. See [Using Recurrent Neural Networks](Feature-Memory.md).                                                                                       | PPO, SAC             |

\*PPO = Proximal Policy Optimization, SAC = Soft Actor-Critic, BC = Behavioral Cloning (Imitation), GAIL = Generative Adversarial Imitaiton Learning

For specific advice on setting hyperparameters based on the type of training you
are conducting, see:

* [Training with PPO](Training-PPO.md)
* [Training with SAC](Training-SAC.md)
* [Using Recurrent Neural Networks](Feature-Memory.md)
* [Training with Curriculum Learning](Training-Curriculum-Learning.md)
* [Training with Imitation Learning](Training-Imitation-Learning.md)
* [Training Generalized Reinforcement Learning Agents](Training-Generalized-Reinforcement-Learning-Agents.md)

You can also compare the
[example environments](Learning-Environment-Examples.md)
to the corresponding sections of the `config/trainer_config.yaml` file for each
example to see how the hyperparameters and other configuration variables have
been changed from the defaults.

### Debugging and Profiling
If you enable the `--debug` flag in the command line, the trainer metrics are logged to a CSV file
stored in the `summaries` directory. The metrics stored are:
  * brain name
  * time to update policy
  * time since start of training
  * time for last experience collection
  * number of experiences used for training
  * mean return

This option is not available currently for Behavioral Cloning.

Additionally, we have included basic [Profiling in Python](Profiling-Python.md) as part of the toolkit.
This information is also saved in the `summaries` directory.
