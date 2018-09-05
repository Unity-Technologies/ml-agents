# Imitation Learning

It is often more intuitive to simply demonstrate the behavior we want an agent
to perform, rather than attempting to have it learn via trial-and-error methods.
Consider our
[running example](ML-Agents-Overview.md#running-example-training-npc-behaviors)
of training a medic NPC : instead of indirectly training a medic with the help
of a reward function, we can give the medic real world examples of observations
from the game and actions from a game controller to guide the medic's behavior.
More specifically, in this mode, the Brain type during training is set to Player
and all the actions performed with the controller (in addition to the agent
observations) will be recorded and sent to the Python API. The imitation
learning algorithm will then use these pairs of observations and actions from
the human player to learn a policy. [Video Link](https://youtu.be/kpb8ZkMBFYs).

## Using Behavioral Cloning

There are a variety of possible imitation learning algorithms which can be used,
the simplest one of them is Behavioral Cloning. It works by collecting training
data from a teacher, and then simply uses it to directly learn a policy, in the
same way the supervised learning for image classification or other traditional
Machine Learning tasks work.

1. In order to use imitation learning in a scene, the first thing you will need
   is to create two Brains, one which will be the "Teacher," and the other which
   will be the "Student." We will assume that the names of the Brain
   `GameObject`s are "Teacher" and "Student" respectively.
2. Set the "Teacher" Brain to Player mode, and properly configure the inputs to
   map to the corresponding actions. **Ensure that "Broadcast" is checked within
   the Brain inspector window.**
3. Set the "Student" Brain to External mode.
4. Link the Brains to the desired Agents (one Agent as the teacher and at least
   one Agent as a student).
5. In `config/trainer_config.yaml`, add an entry for the "Student" Brain. Set
   the `trainer` parameter of this entry to `imitation`, and the
   `brain_to_imitate` parameter to the name of the teacher Brain: "Teacher".
   Additionally, set `batches_per_epoch`, which controls how much training to do
   each moment. Increase the `max_steps` option if you'd like to keep training
   the Agents for a longer period of time.
6. Launch the training process with `mlagents-learn config/trainer_config.yaml
   --train --slow`, and press the :arrow_forward: button in Unity when the
   message _"Start training by pressing the Play button in the Unity Editor"_ is
   displayed on the screen
7. From the Unity window, control the Agent with the Teacher Brain by providing
   "teacher demonstrations" of the behavior you would like to see.
8. Watch as the Agent(s) with the student Brain attached begin to behave
   similarly to the demonstrations.
9. Once the Student Agents are exhibiting the desired behavior, end the training
   process with `CTL+C` from the command line.
10. Move the resulting `*.bytes` file into the `TFModels` subdirectory of the
    Assets folder (or a subdirectory within Assets of your choosing) , and use
    with `Internal` Brain.

### BC Teacher Helper

We provide a convenience utility, `BC Teacher Helper` component that you can add
to the Teacher Agent.

<p align="center">
  <img src="images/bc_teacher_helper.png"
       alt="BC Teacher Helper"
       width="375" border="10" />
</p>

This utility enables you to use keyboard shortcuts to do the following:

1. To start and stop recording experiences. This is useful in case you'd like to
   interact with the game _but not have the agents learn from these
   interactions_. The default command to toggle this is to press `R` on the
   keyboard.

2. Reset the training buffer. This enables you to instruct the agents to forget
   their buffer of recent experiences. This is useful if you'd like to get them
   to quickly learn a new behavior. The default command to reset the buffer is
   to press `C` on the keyboard.
