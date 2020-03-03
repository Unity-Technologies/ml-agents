using UnityEngine;

namespace MLAgents.Sensors
{
    /// <summary>
    /// Editor components for creating Sensors. Generally an ISensor implementation should have a
    /// corresponding SensorComponent to create it.
    /// </summary>
    public abstract class SensorComponent : MonoBehaviour
    {
        /// <summary>
        /// Create the ISensor. This is called by the Agent when it is initialized.
        /// </summary>
        /// <returns>Created ISensor object.</returns>
        public abstract ISensor CreateSensor();

        /// <summary>
        /// Returns the shape of the sensor observations that will be created.
        /// </summary>
        /// <returns>Shape of the sensor observation.</returns>
        public abstract int[] GetObservationShape();

        /// <summary>
        /// Whether the observation is visual or not.
        /// </summary>
        /// <returns>True if the observation is visual, false otherwise.</returns>
        public virtual bool IsVisual()
        {
            var shape = GetObservationShape();
            return shape.Length == 3;
        }

        /// <summary>
        /// Whether the observation is vector or not.
        /// </summary>
        /// <returns>True if the observation is vector, false otherwise.</returns>
        public virtual bool IsVector()
        {
            var shape = GetObservationShape();
            return shape.Length == 1;
        }
    }
}
