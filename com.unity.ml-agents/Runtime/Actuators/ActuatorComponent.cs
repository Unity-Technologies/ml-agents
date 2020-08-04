using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Editor components for creating Actuators. Generally an IActuator component should
    /// have a corresponding ActuatorComponent.
    /// </summary>
    internal abstract class ActuatorComponent : MonoBehaviour
    {
        /// <summary>
        /// Create the IActuator.  This is called by the Agent when it is initialized.
        /// </summary>
        /// <returns>Created IActuator object.</returns>
        public abstract IActuator CreateActuator();
    }
}
