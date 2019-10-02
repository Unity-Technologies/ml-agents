# Training with Imitation Learning

It is often more intuitive to simply demonstrate the behavior we want an agent
to perform, rather than attempting to have it learn via trial-and-error methods.
Consider our
[running example](ML-Agents-Overview.md#running-example-training-npc-behaviors)
of training a medic NPC. Instead of indirectly training a medic with the help
of a reward function, we can give the medic real world examples of observations
from the game and actions from a game controller to guide the medic's behavior.
Imitation Learning uses pairs of observations and actions from
a demonstration to learn a policy. [Video Link](https://youtu.be/kpb8ZkMBFYs).

Imitation learning can also be used to help reinforcement learning. Especially in
environments with sparse (i.e., infrequent or rare) rewards, the agent may never see
the reward and thus not learn from it. Curiosity (which is available in the toolkit)
helps the agent explore, but in some cases
it is easier to show the agent how to achieve the reward. In these cases,
imitation learning combined with reinforcement learning can dramatically
reduce the time the agent takes to solve the environment.
For instance, on the [Pyramids environment](Learning-Environment-Examples.md#pyramids),
using 6 episodes of demonstrations can reduce training steps by more than 4 times.
See PreTraining + GAIL + Curiosity + RL below.

<p align="center">
  <img src="images/mlagents-ImitationAndRL.png"
       alt="Using Demonstrations with Reinforcement Learning"
       width="700" border="0" />
</p>

The ML-Agents toolkit provides several ways to learn from demonstrations.

* To train using GAIL (Generative Adversarial Imitation Learning) you can add the
  [GAIL reward signal](Reward-Signals.md#gail-reward-signal). GAIL can be
  used with or without environment rewards, and works well when there are a limited
  number of demonstrations.
* To help bootstrap reinforcement learning, you can enable
  [pretraining](Training-PPO.md#optional-pretraining-using-demonstrations)
  on the PPO trainer, in addition to using a small GAIL reward signal.
* To train an agent to exactly mimic demonstrations, you can use the
  [Behavioral Cloning](Training-Behavioral-Cloning.md) trainer. Behavioral Cloning can be
  used with demonstrations (in-editor), and learns very quickly. However, it usually is ineffective
  on more complex environments without a large number of demonstrations.

### How to Choose

If you want to help your agents learn (especially with environments that have sparse rewards)
using pre-recorded demonstrations, you can generally enable both GAIL and Pretraining.
An example of this is provided for the Pyramids example environment under
 `PyramidsLearning` in `config/gail_config.yaml`.

If you want to train purely from demonstrations, GAIL is generally the preferred approach, especially
if you have few (<10) episodes of demonstrations. An example of this is provided for the Crawler example
environment under `CrawlerStaticLearning` in `config/gail_config.yaml`.

If you have plenty of demonstrations and/or a very simple environment, Offline Behavioral Cloning can be effective and quick. However, it cannot be combined with RL.

## Recording Demonstrations

It is possible to record demonstrations of agent behavior from the Unity Editor,
and save them as assets. These demonstrations contain information on the
observations, actions, and rewards for a given agent during the recording session.
They can be managed from the Editor, as well as used for training with Offline
Behavioral Cloning and GAIL.

In order to record demonstrations from an agent, add the `Demonstration Recorder`
component to a GameObject in the scene which contains an `Agent` component.
Once added, it is possible to name the demonstration that will be recorded
from the agent.

<p align="center">
  <img src="images/demo_component.png"
       alt="BC Teacher Helper"
       width="375" border="10" />
</p>

When `Record` is checked, a demonstration will be created whenever the scene
is played from the Editor. Depending on the complexity of the task, anywhere
from a few minutes or a few hours of demonstration data may be necessary to
be useful for imitation learning. When you have recorded enough data, end
the Editor play session, and a `.demo` file will be created in the
`Assets/Demonstrations` folder. This file contains the demonstrations.
Clicking on the file will provide metadata about the demonstration in the
inspector.

<p align="center">
  <img src="images/demo_inspector.png"
       alt="BC Teacher Helper"
       width="375" border="10" />
</p>
