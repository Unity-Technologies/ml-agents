# Integration of the Input System Package with ML-Agents

## Overview
One area we are always trying to improve is getting developers up and running with ML-Agents.  With this in mind,
we have implemented an `InputActuatorComponent`.  This component integrates with the
[Input System Package](https://docs.unity3d.com/Packages/com.unity.inputsystem@1.1/manual/QuickStartGuide.html)
to set up an action space for your `Agent` based on an `InputActionAsset` that is referenced by the
`IInputActionAssetProvider` interface, or the `PlayerInput` component that may be living on your player controlled
`Agent`.  This means that if you have code outside of your agent that handles input, you will not need to implement
the Heuristic function in agent as well.  The `InputActuatorComponent` will handle this for you.  You can now train and
run inference on `Agents` with an action space defined by an `InputActionAsset`.

This implementation includes:

* C# `InputActuatorComponent` you can attach to your Agent.
* Implement the `IInputActionAssetProvider` in the `Componenet` where you handle player input.
* An example environment where the input handling code is not in the Heuristic function of the Agent subclass.

### Feedback
We have only implemented a small subset of `InputControl` types that we thought would cover a large portion of what
most developers would use.  Please let us know if you want more control types implemented by posting in the [ML-Agents
forum.](https://forum.unity.com/forums/ml-agents.453/)

We would also like your feedback on the workflow of integrating this into your games.  If you run
into workflow issues please let us know in the ML-Agents forums, or if you've discovered a bug,
please file a bug on our GitHub page.

## Getting started
The C# code for the `InputActuatorComponent` exists inside of the extensions package (com.unity.ml-agents.extensions).  A good first step would be to familiarize with the extensions package by reading the document [here](com.unity.ml-agents.extensions.md).  The second step would be to take a look at how we have implemented the C# code in the example Input Integration scene (located under  ML-Agents-Input-Example/Assets/ML-Agents/Examples/PushBlock/).  Once you have some familiarity, then the next step would be to add the InputActuatorComponent to your player Agent.  The example we have implemented uses C# Events to send information from the Input System.

Additionally, see below for additional technical specifications on the C# code for the InputActuatorComponent.

## Technical specifications for the InputActuatorComponent

### `IInputActionsAssetProvider` Interface
The `InputActuatorComponent` searches for a `Component` that implements
`IInputActionAssetProvider` on the `GameObject` they both are attached to.  It is important to note
that if multiple `Components` on your `GameObject` need to access an `InputActionAsset` to handle events,
they will need to share the same instance of the `InputActionAsset` that is returned from the
`IInputActionAssetProvider`.

### `InputActuatorComponent` class
The `InputActuatorComponent` is the bridge between ML-Agents and the Input System.. It allows ML-Agents to
* create an `ActionSpec` for your Agent based on an `InputActionAsset` that comes from an
`IInputActionAssetProvider`.
* send simulated input from a training process or a neural network
* let developers keep their input handling code in one place

This is accomplished by adding the `InputActuatorComponenet` to an Agent which already has the PlayerInput component attached.

### Setting up a scene using the `InputActuatorComponent`
1. Add the `com.unity.inputsystem` version 1.1.0-preview.3 or later to your project via the Package Manager window.
2. If you have already setup an InputActionAsset skip to Step 3, otherwise follow these sub steps:
    1. Create an InputActionAsset to allow your Agent to be controlled by the Input System.
    2. Handle the events from the Input System where you normally would (i.e. a script external to your Agent class).
3. Add the InputSystemActuatorComponent to the GameObject that has the `PlayerInput` and `Agent` components attached.

