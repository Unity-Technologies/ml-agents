# Learning Brains

The **Learning Brain** works differently if you are training it or not.
When training your Agents, drag the **Learning Brain** to the
Academy's `Broadcast Hub` and check the checkbox `Control`. When using a pre-trained 
model, just drag the Model file into the `Model` property of the **Learning Brain**.

## Training Mode / External Control

When [running an ML-Agents training algorithm](Training-ML-Agents.md), at least
one Brain asset must be in the Academy's `Broadcast Hub` with the checkbox `Control`
checked. This allows the training process to collect the observations of Agents 
using that Brain and give the Agents their actions.

In addition to using a **Learning Brain** for training using the ML-Agents learning
algorithms, you can use a **Learning Brain** to control Agents in a Unity
environment using an external Python program. See [Python API](Python-API.md)
for more information.

## Inference Mode / Internal Control

When not training, the **Learning Brain** uses a
[TensorFlow model](https://www.tensorflow.org/get_started/get_started_for_beginners#models_and_training)
to make decisions. The Proximal Policy Optimization (PPO) and Behavioral Cloning
algorithms included with the ML-Agents SDK produce trained TensorFlow models
that you can use with the Learning Brain type.

A __model__ is a mathematical relationship mapping an agent's observations to
its actions. TensorFlow is a software library for performing numerical
computation through data flow graphs. A TensorFlow model, then, defines the
mathematical relationship between your Agent's observations and its actions
using a TensorFlow data flow graph.

### Creating a graph model

The training algorithms included in the ML-Agents SDK produce TensorFlow graph
models as the end result of the training process. See
[Training ML-Agents](Training-ML-Agents.md) for instructions on how to train a
model.

### Using a graph model

To use a graph model:

1. Select the **Learning Brain** asset in the **Project** window of the Unity Editor.
2. Import the `model_name` file produced by the PPO training
   program. (Where `model_name` is the name of the model file, which is
   constructed from the name of your Unity environment executable and the run-id
   value you assigned when running the training process.)

   You can
   [import assets into Unity](https://docs.unity3d.com/Manual/ImportingAssets.html)
   in various ways. The easiest way is to simply drag the file into the
   **Project** window and drop it into an appropriate folder.
3. Once the `model_name.nn` file is imported, drag it from the **Project**
   window to the **Model** field of the Brain component.

If you are using a model produced by the ML-Agents `mlagents-learn` command, use
the default values for the other Learning Brain parameters.

### Learning Brain properties

The default values of the TensorFlow graph parameters work with the model
produced by the PPO and BC training code in the ML-Agents SDK. To use a default
ML-Agents model, the only parameter that you need to set is the `Model`,
which must be set to the `.nn` file containing the trained model itself.

* `Model` : This must be the `.nn` file corresponding to the pre-trained
   TensorFlow graph. (You must first drag this file into your Project window 
   and then from the Resources folder into the inspector)

