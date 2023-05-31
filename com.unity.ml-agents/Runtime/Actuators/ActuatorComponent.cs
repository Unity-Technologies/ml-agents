using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Editor components for creating Actuators. Generally an IActuator component should
    /// have a corresponding ActuatorComponent.
    /// </summary>
    public abstract class ActuatorComponent : MonoBehaviour
    {
        /// <summary>
        /// Create a collection of <see cref="IActuator"/>s.  This is called by the <see cref="Agent"/> during
        /// initialization.
        /// </summary>
        /// <returns>A collection of <see cref="IActuator"/>s</returns>
        public abstract IActuator[] CreateActuators();

        /// <summary>
        /// The specification of the possible actions for this ActuatorComponent.
        /// This must produce the same results as the corresponding IActuator's ActionSpec.
        /// </summary>
        /// <seealso cref="ActionSpec"/>
        public abstract ActionSpec ActionSpec { get; }
    }
}
