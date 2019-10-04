# Brains

The Brain encapsulates the decision making process. Every Agent must be
assigned a Brain, but you can use the same Brain with more than one Agent. You
can also create several Brains, attach each of the Brain to one or more than one
Agent.

There are 3 kinds of Brains you can use:

* [Learning](Learning-Environment-Design-Learning-Brains.md) – Use a
  **LearningBrain** to make use of a trained model or train a new model.
* [Heuristic](Learning-Environment-Design-Heuristic-Brains.md) – Use a
  **HeuristicBrain** to hand-code the Agent's logic by extending the Decision class.
* [Player](Learning-Environment-Design-Player-Brains.md) – Use a
   **PlayerBrain** to map keyboard keys to Agent actions, which can be 
   useful to test your Agent code.

During training, use a **Learning Brain**.
When you want to use the trained model, import the model file into the Unity
project, add it to the **Model** property of the **Learning Brain**.

Brain assets has several important properties that you can set using the
Inspector window. These properties must be appropriate for the Agents using the
Brain. For example, the `Vector Observation Space Size` property must match the
length of the feature vector created by an Agent exactly. See
[Agents](Learning-Environment-Design-Agents.md) for information about creating
agents and setting up a Brain instance correctly.

## Brain Properties

The Brain Inspector window in the Unity Editor displays the properties assigned
to a Brain component:

![Brain Inspector](images/brain.png)

* `Brain Parameters` - Define vector observations, visual observation, and
  vector actions for the Brain.
  * `Vector Observation`
    * `Space Size` - Length of vector observation for Brain.
    * `Stacked Vectors` - The number of previous vector observations that will
      be stacked and used collectively for decision making. This results in the
      effective size of the vector observation being passed to the Brain being:
      _Space Size_ x _Stacked Vectors_.
  * `Visual Observations` - Describes height, width, and whether to grayscale
    visual observations for the Brain.
  * `Vector Action`
    * `Space Type` - Corresponds to whether action vector contains a single
      integer (Discrete) or a series of real-valued floats (Continuous).
    * `Space Size` (Continuous) - Length of action vector for Brain.
    * `Branches` (Discrete) - An array of integers, defines multiple concurrent
      discrete actions. The values in the `Branches` array correspond to the
      number of possible discrete values for each action branch.
    * `Action Descriptions` - A list of strings used to name the available
      actions for the Brain.

The other properties of the Brain depend on the type of Brain you are using.
