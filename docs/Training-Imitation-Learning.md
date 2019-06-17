# Imitation Learning

It is often more intuitive to simply demonstrate the behavior we want an agent
to perform, rather than attempting to have it learn via trial-and-error methods.
Consider our
[running example](ML-Agents-Overview.md#running-example-training-npc-behaviors)
of training a medic NPC. Instead of indirectly training a medic with the help
of a reward function, we can give the medic real world examples of observations
from the game and actions from a game controller to guide the medic's behavior.
Imitation Learning uses pairs of observations and actions from
from a demonstration to learn a policy. [Video Link](https://youtu.be/kpb8ZkMBFYs).

ML-Agents provides several ways to interact with demonstrations. 

## Recording Demonstrations

It is possible to record demonstrations of agent behavior from the Unity Editor, 
and save them as assets. These demonstrations contain information on the 
observations, actions, and rewards for a given agent during the recording session. 
They can be managed from the Editor, as well as used for training with Offline 
Behavioral Cloning (see below).

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
 