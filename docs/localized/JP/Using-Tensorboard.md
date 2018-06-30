# Using TensorBoard to Observe Training

The ML-Agents toolkit saves statistics during learning session that you can view with a TensorFlow utility named, [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).

The `learn.py` program saves training statistics to a folder named `summaries`, organized by the `run-id` value you assign to a training session.

In order to observe the training process, either during training or afterward, 
start TensorBoard:

1. Open a terminal or console window:
2. Navigate to the ml-agents/python folder.
3. From the command line run :

        tensorboard --logdir=summaries

4. Open a browser window and navigate to [localhost:6006](http://localhost:6006).

**Note:** If you don't assign a `run-id` identifier, `learn.py` uses the default string, "ppo". All the statistics will be saved to the same sub-folder and displayed as one session in TensorBoard. After a few runs, the displays can become difficult to interpret in this situation. You can delete the folders under the `summaries` directory to clear out old statistics.

On the left side of the TensorBoard window, you can select which of the training runs you want to display. You can select multiple run-ids to compare statistics. The TensorBoard window also provides options for how to display and smooth graphs.
 
When you run the training program, `learn.py`, you can use the `--save-freq` option to specify how frequently to save the statistics.

## The ML-Agents toolkit training statistics

The ML-agents training program saves the following statistics:

![Example TensorBoard Run](images/mlagents-TensorBoard.png)

* Lesson - Plots the progress from lesson to lesson. Only interesting when performing
[curriculum training](Training-Curriculum-Learning.md). 

* Cumulative Reward - The mean cumulative episode reward over all agents. 
Should increase during a successful training session.

* Entropy - How random the decisions of the model are. Should slowly decrease 
during a successful training process. If it decreases too quickly, the `beta` 
hyperparameter should be increased.

* Episode Length - The mean length of each episode in the environment for all 
agents.

* Learning Rate - How large a step the training algorithm takes as it searches 
for the optimal policy. Should decrease over time.

* Policy Loss - The mean magnitude of policy loss function. Correlates to how
much the policy (process for deciding actions) is changing. The magnitude of 
this should decrease during a successful training session.

* Value Estimate - The mean value estimate for all states visited by the agent. 
Should increase during a successful training session.

* Value Loss - The mean loss of the value function update. Correlates to how
well the model is able to predict the value of each state. This should increase
while the agent is learning, and then decrease once the reward stabilizes.

* _(Curiosity-Specific)_ Intrinsic Reward - This corresponds to the mean cumulative intrinsic reward generated per-episode. 

* _(Curiosity-Specific)_ Forward Loss - The mean magnitude of the inverse model loss function. Corresponds to how well the model is able to predict the new observation encoding.

* _(Curiosity-Specific)_ Inverse Loss - The mean magnitude of the forward model loss function. Corresponds to how well the model is able to predict the action taken between two observations.