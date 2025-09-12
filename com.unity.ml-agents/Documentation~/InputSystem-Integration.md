# Input System Integration

The ML-Agents package integrates with the [Input System Package](https://docs.unity3d.com/Packages/com.unity.inputsystem@1.14/manual/QuickStartGuide.html) through the `InputActuatorComponent`. This component sets up an action space for your `Agent` based on an `InputActionAsset` that is referenced by the `IInputActionAssetProvider` interface, or the `PlayerInput` component that may be living on your player controlled `Agent`. This means that if you have code outside of your agent that handles input, you will not need to implement the Heuristic function in agent as well. The `InputActuatorComponent` will handle this for you. You can now train and run inference on `Agents` with an action space defined by an `InputActionAsset`.

Take a look at how we have implemented the C# code in the example Input Integration scene (located under Project/Assets/ML-Agents/Examples/PushBlockWithInput/). Once you have some familiarity, then the next step would be to add the InputActuatorComponent to your player Agent. The example we have implemented uses C# Events to send information from the Input System.

## Getting Started with Input System Integration
1. Add the `com.unity.inputsystem` version 1.1.0-preview.3 or later to your project via the Package Manager window.
2. If you have already setup an InputActionAsset skip to Step 3, otherwise follow these sub steps:
3. Create an InputActionAsset to allow your Agent to be controlled by the Input System.
4. Handle the events from the Input System where you normally would (i.e. a script external to your Agent class).
5. Add the InputSystemActuatorComponent to the GameObject that has the `PlayerInput` and `Agent` components attached.

Additionally, see below for additional technical specifications on the C# code for the InputActuatorComponent.
## Technical Specifications

### `IInputActionsAssetProvider` Interface
The `InputActuatorComponent` searches for a `Component` that implements
`IInputActionAssetProvider` on the `GameObject` they both are attached to. It is important to note
that if multiple `Components` on your `GameObject` need to access an `InputActionAsset` to handle events, they will need to share the same instance of the `InputActionAsset` that is returned from the
`IInputActionAssetProvider`.

### `InputActuatorComponent` Class
The `InputActuatorComponent` is the bridge between ML-Agents and the Input System. It allows ML-Agents to:
* create an `ActionSpec` for your Agent based on an `InputActionAsset` that comes from an `IInputActionAssetProvider`.
* send simulated input from a training process or a neural network
* let developers keep their input handling code in one place

This is accomplished by adding the `InputActuatorComponent` to an Agent which already has the PlayerInput component attached.

## Requirements

If using the `InputActuatorComponent`, install the `com.unity.inputsystem` package `version 1.1.0-preview.3` or later.

## Known Limitations

For the `InputActuatorComponent`
- Limited implementation of `InputControls`
- No way to customize the action space of the `InputActuatorComponent`
