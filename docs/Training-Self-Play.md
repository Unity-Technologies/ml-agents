# Training with Self-Play

ML-Agents provides the functionality to train symmetric, adversarial games with [Self-Play](https://openai.com/blog/competitive-self-play/).
With self-play, agents learn in adversarial games by competing against past versions of itself. Competing
against fixed, past versions of the agent's policy provides a more stable, stationary learning environment. This is compared
to competing against its current self in every episode, which is a constantly changing opponent.

Self-play can be used with our implementations of both [Proximal Policy Optimization (PPO)](Training-PPO.md) and [Soft Actor-Critc (SAC)](Training-SAC.md).
For more general information on training with ML-Agents, see [Training ML-Agents](Training-ML-Agents.md).
For more algorithm specific instruction, please see the documentation for [PPO](Training-PPO.md) or [SAC](Training-SAC.md).

Self-play is triggered by including the self-play hyperparameter hierarchy in the trainer configuration file.  Detailed description of the self-play hyperparameters are contained below. Furthermore, to distinguish opposing agents, set the team ID to different integer values in the behavior parameters script on the agent prefab.

![Team ID](images/team_id.png)

See the trainer configuration and agent prefabs for our Tennis example environment for an example.

## Training with self-play

Training with self-play adds additional confounding factors to the usual issues faced by reinforcement learning in general. This guide contains some discussion of the self-play hyperparameters and intuitions for tuning them.

The general tradeoff is between speed/generality/stability of learning
## Hyperparameters

### Reward Signals

The reward signal should still be used as described in the documentation for the other trainers and [reward signals.](Reward-Signals.md) However, we encourage users to be a bit more conservative when shaping reward functions due to the instability and non-stationarity of learning in adversarial games. Specifically, we encouraging users to begin with the simplest possible reward function (+1 winning, -1 losing) and to allow for more iterations of training to compensate for the sparsity.

### Play against current self ratio

The `play_against_current_self_ratio` parameter corresponds to the probability
an agent will play against its ***current*** self. With probability `1 - play_against_current_self_ratio`, the agent will play against a snapshot of itself from a past iteration.

### Window

The `window` parameter corresponds to the size of the sliding window of past snapshots from which the agent's opponents are sampled.  For example, a `window` size of 5 will save the last 5 snapshots taken. Each time a new snapshot is taken, the oldest is discarded.

### Snapshot Per

The `snapshot_per` parameter corresponds to the number of *trainer steps* between snapshots.  For example, if `snapshot_per=10000` then a snapshot of the current policy will be saved every 10000 trainer steps. Note, trainer steps are counted per agent. For more information, please see the [migration doc](Migrating.md) after v0.13.

## Training Statistics

To view training statistics, use TensorBoard. For information on launching and
using TensorBoard, see
[here](./Getting-Started-with-Balance-Ball.md#observing-training-progress).

### ELO
In adversarial games, the cumulative environment reward may not be a meaningful metric by which to track learning progress.  This is because cumulative reward is entirely dependent on the skill of the opponent. An agent at a particular skill level will get more or less reward against a worse or better agent, respectively.

We provide an implementation of the ELO rating system, a method for calculating the relative skill level between two players from a given population in a zero-sum game. For more informtion on ELO, please see [the ELO wiki](https://en.wikipedia.org/wiki/Elo_rating_system).
