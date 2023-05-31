#if MLA_INPUT_SYSTEM
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Input
{
    /// <summary>
    /// Implement this interface if you are listening to C# events from the generated C# class from the
    /// <see cref="InputActionAsset"/>. This interface works with the <see cref="InputActuatorComponent"/> in order
    /// to allow ML-Agents to simulate input actions based on the instance of the <see cref="InputActionAsset"/>
    /// used to listen to events.  If you implement this interface the <see cref="InputActuatorComponent"/> will use
    /// what is returned from <see cref="GetInputActionAsset"/> as the asset to base it's simulated input for.
    /// Otherwise, the <see cref="InputActuatorComponent"/> will look for the <see cref="PlayerInput"/> component
    /// and use the asset from there.  If you have multiple components handling PlayerInput on the same GameObject
    /// they will need to share the same instance of the <see cref="InputActionAsset"/> in order to get the simulated
    /// input.
    /// </summary>
    public interface IInputActionAssetProvider
    {
        /// <summary>
        /// Returns the <see cref="InputActionAsset"/> instance being from the generated C# class of the
        /// <see cref="InputActionAsset"/> in order to correctly fire events when simulating input from ML-Agents.
        /// </summary>
        /// <returns>The instance of the <see cref="InputActionAsset"/> you are listening for events on.</returns>
        (InputActionAsset, IInputActionCollection2) GetInputActionAsset();
    }
}
#endif // MLA_INPUT_SYSTEM
