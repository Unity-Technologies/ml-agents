# Using TensorBoard to Observe Training

The ML-Agents Toolkit saves statistics during learning session that you can view
with a TensorFlow utility named,
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).

The `mlagents-learn` command saves training statistics to a folder named
`summaries`, organized by the `run-id` value you assign to a training session.

In order to observe the training process, either during training or afterward,
start TensorBoard:

1. Open a terminal or console window:
1. Navigate to the directory where the ML-Agents Toolkit is installed.
1. From the command line run: `tensorboard --logdir=summaries --port=6006`
1. Open a browser window and navigate to
   [localhost:6006](http://localhost:6006).

**Note:** The default port TensorBoard uses is 6006. If there is an existing
session running on port 6006 a new session can be launched on an open port using
the --port option.

**Note:** If you don't assign a `run-id` identifier, `mlagents-learn` uses the
default string, "ppo". All the statistics will be saved to the same sub-folder
and displayed as one session in TensorBoard. After a few runs, the displays can
become difficult to interpret in this situation. You can delete the folders
under the `summaries` directory to clear out old statistics.

On the left side of the TensorBoard window, you can select which of the training
runs you want to display. You can select multiple run-ids to compare statistics.
The TensorBoard window also provides options for how to display and smooth
graphs.

When you run the training program, `mlagents-learn`, you can use the
`--save-freq` option to specify how frequently to save the statistics.

## The ML-Agents Toolkit training statistics

The ML-Agents training program saves the following statistics:

![Example TensorBoard Run](images/mlagents-TensorBoard.png)

### Environment Statistics

- `Environment/Lesson` - Plots the progress from lesson to lesson. Only
  interesting when performing curriculum training.

- `Environment/Cumulative Reward` - The mean cumulative episode reward over all
  agents. Should increase during a successful training session.

- `Environment/Episode Length` - The mean length of each episode in the
  environment for all agents.

### Policy Statistics

- `Policy/Entropy` (PPO; SAC; BC) - How random the decisions of the model are. Should
  slowly decrease during a successful training process. If it decreases too
  quickly, the `beta` hyperparameter should be increased.

- `Policy/Learning Rate` (PPO; SAC; BC) - How large a step the training algorithm
  takes as it searches for the optimal policy. Should decrease over time.

- `Policy/Value Estimate` (PPO; SAC) - The mean value estimate for all states visited
  by the agent. Should increase during a successful training session.

- `Policy/Curiosity Reward` (PPO/SAC+Curiosity) - This corresponds to the mean
  cumulative intrinsic reward generated per-episode.

- `Policy/Entropy Coefficient` (SAC) - The entropy coefficient balances the incentive to behave randomly vs. maximizing the reward.
This value is adjusted automatically so that the agent retains some amount of randomness during
training. It should steadily decrease in the beginning of training, and reach some small
value where it will level off. If it decreases too soon or takes too
long to decrease, `init_entcoef` should be adjusted.


### Learning Loss Functions

- `Losses/Policy Loss` (PPO; SAC) - The mean magnitude of policy loss function.
  Correlates to how much the policy (process for deciding actions) is changing.
  The magnitude of this should decrease during a successful training session.

- `Losses/Value Loss` (PPO; SAC) - The mean loss of the value function update.
  Correlates to how well the model is able to predict the value of each state.
  This should increase while the agent is learning, and then decrease once the
  reward stabilizes.

- `Losses/Forward Loss` (PPO/SAC+Curiosity) - The mean magnitude of the inverse
  model loss function. Corresponds to how well the model is able to predict the
  new observation encoding.

- `Losses/Inverse Loss` (PPO/SAC+Curiosity) - The mean magnitude of the forward
  model loss function. Corresponds to how well the model is able to predict the
  action taken between two observations.

- `Losses/Cloning Loss` (BC) - The mean magnitude of the behavioral cloning
  loss. Corresponds to how well the model imitates the demonstration data.

### Self-Play

- `Self-Play/ELO` (Self-Play) - The [ELO](https://en.wikipedia.org/wiki/Elo_rating_system) rating system is a method for calculating the relative skill level between two players from a given population in a zero-sum game.
In a proper training run, the ELO of the agent should steadily increase. The absolute value of the ELO is less important than the change in ELO over training iterations.


## Custom Metrics from Unity

To get custom metrics from a C# environment into Tensorboard, you can use the
`StatsRecorder`:

```csharp
var statsRecorder = Academy.Instance.StatsRecorder;
statsSideChannel.Add("MyMetric", 1.0);
```
