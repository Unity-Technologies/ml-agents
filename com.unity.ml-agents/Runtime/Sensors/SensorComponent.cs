using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Editor components for creating Sensors. Generally an ISensor implementation should have a
    /// corresponding SensorComponent to create it.
    /// </summary>
    public abstract class SensorComponent : MonoBehaviour
    {
        /// <summary>
        /// Create the ISensors. This is called by the Agent when it is initialized.
        /// </summary>
        /// <returns>Created ISensor objects.</returns>
        public abstract ISensor[] CreateSensors();
    }
}
