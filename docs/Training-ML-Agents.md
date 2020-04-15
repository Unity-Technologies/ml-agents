# Training ML-Agents

For a broad overview of reinforcement learning, imitation learning and all the
training scenarios, methods and options within the ML-Agents Toolkit, see
[ML-Agents Toolkit Overview](ML-Agents-Overview.md).

Once your learning environment has been created and is ready for training, the next
step is to initiate a training run. Training in the ML-Agents Toolkit is powered
by a dedicated Python package, `mlagents`. This package exposes a command `mlagents-learn` that
is the single entry point for all training workflows (e.g. reinforcement
leaning, imitation learning, curriculum learning). Its implementation can be found at
[ml-agents/mlagents/trainers/learn.py](../ml-agents/mlagents/trainers/learn.py).

## Training with mlagents-learn

### Starting Training

`mlagents-learn` is the main training utility provided by the ML-Agents Toolkit. It
accepts a number of CLI options in addition to a YAML configuration file that contains
all the configurations and hyperparameters to be used during training. The set of
configurations and hyperparameters to include in this file depend on the agents in your
environment and the specific training method you wish to utilize. Keep in mind that
the hyperparameter values can have a big impact on the training performance (i.e. your
agent's ability to learn a policy that solves the task). In this page, we will review all the
hyperparameters for all training methods and provide guidelines and advice on their values.

To view a description of all the CLI options accepted by `mlagents-learn`, use the `--help`:
```sh
mlagents-learn --help
```

The basic command for training is:

```sh
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>
```

where

* `<trainer-config-file>` is the file path of the trainer configuration yaml. This contains all the
  hyperparameter values. We offer a detailed guide on the structure of this file and the meaning
  of the hyperameters (and advice on how to set them) in the dedicated
  [Training Config File](#training-config-file) section below.
* `<env_name>`__(Optional)__ is the name (including path) of your [Unity
  executable](Learning-Environment-Executable.md) containing the agents to be trained.
  If `<env_name>` is not passed, the training will happen in the Editor.
  Press the :arrow_forward: button in Unity when the message _"Start training by
  pressing the Play button in the Unity Editor"_ is displayed on the screen.
* `<run-identifier>` is a unique name you can use to identify the results of your training runs.

See the [Getting Started Guide](Getting-Started.md#training-a-new-model-with-reinforcement-learning)
for a sample execution of the `mlagents-learn` command.

#### Observing Training

Regardless of which training methods, configurations or hyperparameters you provide,
the training process will always generate three artifacts:
1. Summaries (under the `summaries/` folder): these are training metrics that are updated
throughout the training process. They are helpful to monitor your training performance
and may help inform how to update your hyperparameter values.
See [Using TensorBoard](Using-Tensorboard.md) for more details on how to visualize
the training metrics.
1. Models (under the `models/` folder): these contain the model checkpoints that are updated
throughout training and the final model file (`.nn`). This final model file is generated once
either when training completes or is interrupted.
1. Timers file (also under the `summaries/` folder): this contains aggregated metrics on your
training process, including time spent on specific code blocks.
See [Profiling in Python](Profiling-Python.md) for more information on the timers generated.

These artifacts (except the `.nn` file) are updated throughout the training process and finalized
when training completes or is interrupted.

#### Stopping and Resuming Training

To interrupt training and save the current progress, hit `Ctrl+C` once and wait for the
model(s) to be saved out.

To resume a previously interrupted or completed training run, use the `--resume` flag and
make sure to specify the previously used run ID.

If you would like to re-run a previously interrupted or completed training run and re-use
the same run ID (in this case, overwriting the previously generated artifacts), then
use the `--force` flag.

#### Loading an Existing Model

You can also use this mode to run inference of an already-trained model in Python by
using both the `--resume` and `--inference` flags. Note that if you want to run
inference in Unity, you should use the [Unity Inference Engine](Getting-Started.md#running-a-pre-trained-model).

Alternatively, you might want to start a new training run but _initialize_ it using an already-trained
model. You may want to do this, for instance, if your environment changed and you want
a new model, but the old behavior is still better than random. You can do this by specifying `--initialize-from=<run-identifier>`, where `<run-identifier>` is the old run ID.

## Training Config File

The Unity ML-Agents Toolkit provides a wide range of training scenarios, methods and options.
As such, specific training runs may require different training configurations and may
generate different artifacts and TensorBoard statistics. This section offers a detailed
guide into how to manage the different training set-ups withing the toolkit.

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

\*PPO = Proximal Policy Optimization, SAC = Soft Actor-Critic, BC = Behavioral Cloning (Imitation), GAIL = Generative Adversarial Imitation Learning

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
| learning_rate_schedule | Determines how learning rate changes over time. | PPO, SAC |
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
| init_path        | Initialize trainer from a previously saved model.                                                                                       | PPO, SAC             |

For specific advice on setting hyperparameters based on the type of training you
are conducting, see:

* [Training with PPO](Training-PPO.md)
* [Training with SAC](Training-SAC.md)
* [Training with Self-Play](Training-Self-Play.md)
* [Using Recurrent Neural Networks](Feature-Memory.md)
* [Training with Curriculum Learning](Training-Curriculum-Learning.md)
* [Training with Imitation Learning](Training-Imitation-Learning.md)
* [Training with Environment Parameter Randomization](Training-Environment-Parameter-Randomization.md)

You can also compare the [example environments](Learning-Environment-Examples.md)
to the corresponding sections of the `config/trainer_config.yaml` file for each
example to see how the hyperparameters and other configuration variables have
been changed from the defaults.
