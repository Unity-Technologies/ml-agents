# Background: Machine Learning

**Work In Progress**

We will not attempt to provide a thorough treatment of machine learning
as there are fantastic resources online. However, given that a number
of users of ML-Agents might not have a formal machine learning background,
the goal of this section is to overview terminologies to facilitate the
understanding of ML-Agents.

Machine learning is a branch of artificial intelligence that is focused
on learning patterns from data. There are three
classes of machine learning algorithms: unsupervised learning, supervised
learning and reinforcement learning. What separates each of these classes is
the type of data available to learn from. In the following paragraphs we
overview each of these classes and provide introductory examples. 

## Unsupervised Learning

The goal in unsupervised learning is to group or cluster a data set. 
For example, consider the players of a game. We may want to group the players
depending on how engaged they are with the game. This could enable us to 
target different groups (e.g. for highly-engaged players we would
invite them to be beta testers for new features, while for unengaged players
we would email them helpful tutorials). Say,
for simplicity, that we wish to split our players into two groups. We would
first define basic attributes of the players, such as number of hours
played, total money spent on in-app purchases and
number of levels completed. We can then feed this data set (three
attributes for every player)
to an unsupervised learning algorithm where we specify the number of groups
to be two. The output would be a split of all the players into two groups,
where one group would semantically represent the engaged players and the second
group would semantically represent the unengaged players. Note that we did
not specify what semantic groups we wanted, but by defining the appropriate
attributes, these semantic groupings are a by-product. We consider this
data set unlabeled because for each player we did not provide any
ground-truth assignment or label. In the next paragraph we overview 
supervised learning, which accepts as input labels in addition to attributes.

## Supervised Learning

In supervised learning we wish to learn an input to output mapping between
the attributes and corresponding labels. Returning to our earlier example of
clustering players, let's say we now wish to predict which of our players are
about to churn (that is stop playing the game for the next 30 days). In this
case, we can look into our historical records and create a data set that
contains attributes of our players in addition to a label indicating whether
they have churned or not. Note that the player attributes we use for this
churn prediction task may be different from the ones we used for our earlier
clustering task. We can then feed this data set (attributes **and** label for
each player) into a supervised
learning algorithm which would learn a mapping from the player attributes
to a binary label indicating whether that player will churn or not.
Now given this learned model, we can provide it the attributes of a
new player (one that recently started playing the game) and it would output
a binary label which serves as a prediction for whether this player will
churn or not. We can now use these predictions to target the players
who are expected to churn and entice them to continue playing the game.

As you may have noticed, for both supervised and unsupervised learning there
are two tasks that need to be performed: attribute selection and model
selection. Attribute selection (also called feature selection) pertains to
selecting how we wish to represent the entity of interest, in this case, the
player. Model selection, on the other hand, pertains to selecting the
algorithm (and its parameters) that perform the task well. Both of these
tasks are active areas of machine learning research and in practice require
several iterations. 

We now switch to reinforcement learning, the third class of
machine learning algorithms, and arguably the one most relevant for ML-Agents.

## Reinforcement Learning

Reinforcement learning can be viewed as a form of learning for sequential
decision making that is typically associated with controlling robots (but is,
in fact, much more general). Consider an autonomous firefighting robot that is
tasked with navigating into an area, finding the fire and neutralizing it. At
every instance the robot perceives the environment through its sensors (e.g.
camera, heat, touch), processes this information and produces an action (e.g.
move to the left, rotate the water hose, turn on the water). In other words,
it is continuously making decisions about how to interact in this environment
given its view of the world (i.e. sensors input) and objective (i.e.
neutralizing the fire). Teaching a robot to be a successful firefighting
machine is precisely what reinforcement learning is designed to do. 

More specifically, the goal of reinforcement learning is to learn a **policy**, 
which is essentially a mapping from **observations** to **actions**. An 
observation is what the robot can measure from its **environment** (in this 
case, all its sensory inputs) and an action, in its most raw form, is a change
to the configuration of the robot (e.g. position of its base, position of
its water hose and whether the hose is on or off). It is common to confuse a
robot's observation with the environment **state**. The environment state 
represents information about all the entities within the environment. The 
robot's observation, however, only contains information that the robot can
measure or perceive about its environment and is typically a subset of the 
environment state. For example, the location of the fire cannot be part of
the robot's observation if it has not even entered the area yet, but it is
part of the environment state.

The last remaining piece
of the reinforcement learning task is the **reward signal**. When training a
robot to be a mean firefighting machine, we provide it with rewards (positive 
and negative) indicating how well it is doing on completing the task.
Note that the objective itself is not fully specified for the robot, but the 
fact that it receives a large positive reward when it puts out the fire and a 
small negative reward for every passing second is all the input the robot
should need to learn that its objective is to put out the fire in the 
shortest amount of time. The fact that rewards are sparse (i.e. may not be
provided at every step, but only when a robot arrives at a success or failure
situation), is a defining characteristic of reinforcement learning and
precisely why learning good policies can be difficult (and/or time-consuming)
for complex environments. 

<p align="center">
  <img src="images/rl_cycle.png" alt="The reinforcement learning cycle."/>
</p>

[Learning a policy](https://blogs.unity3d.com/2017/08/22/unity-ai-reinforcement-learning-with-q-learning/)
is usually achieved through many trials and iterative
policy updates. More specifically, the robot will be placed in several
fire situations and over time will learn an optimal policy which allows it
to put our fires more effectively. Obviously, we cannot expect to train a
robot repeatedly in the real world, particularly when fires are involved. This
is precisely why the use of 
[Unity as a simulator](https://blogs.unity3d.com/2018/01/23/designing-safer-cities-through-simulations/)
serves as the perfect training grounds for learning such behaviors.
While our discussion of reinforcement learning has centered around robots,
there are strong parallels between robots and characters in a game. In fact,
in many ways, one can view a non-playable character (NPC) as a virtual
robot, with its own observations about the environment, its own set of actions
and a specific objective. Thus it is natural to explore how we can
train behaviors within Unity using reinforcement learning. This is precisely
what ML-Agents offers. The video linked below includes a reinforcement
learning demo showcasing training character behaviors using ML-Agents.

<p align="center">
    <a href="http://www.youtube.com/watch?feature=player_embedded&v=fiQsmdwEGT8" target="_blank">
        <img src="http://img.youtube.com/vi/fiQsmdwEGT8/0.jpg" alt="RL Demo" width="400" border="10" />
    </a>
</p>

Similar to both unsupervised and supervised learning, reinforcement learning
is also faced with two tasks: attribute selection and model selection.
Attribute selection is defining the set of observations for the robot
that best help it complete its objective, while model selection is defining
the form of the policy (mapping from observations to actions) and its
parameters. In practice, training behaviors is an iterative process that may
require changing the attribute and model choices.

## Training and Inference

One common aspect of all three branches of machine learning is that they
both include a **training phase** and an **inference phase**. While the
details of the training and inference phases are different for each of the
three classes, at a high-level, the training phase involves building a model
using the provided data, while the inference phase involves applying this
model to new, previously unseen, data. From our unsupervised learning
example above, the training phase is the one where we cluster the players
into the two groups, while the inference phase is when we assign a new player 
to one of the two clusters. From our supervised learning example, the 
training phase is when the mapping from player attributes to player label
(whether they churned or not) is learned, and the inference phase is
when that learned mapping is used to predict whether a new player will churn
or not. Lastly, for reinforcement learning, the training phase is when the
policy is learned and the inference phase is when the agent starts
interacting in the wild using its learned policy.

## Deep Learning

To be completed.

Link to TensorFlow background page.
