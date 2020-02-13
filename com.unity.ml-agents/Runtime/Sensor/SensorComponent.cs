using System;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Editor components for creating Sensors. Generally an ISensor implementation should have a corresponding
    /// SensorComponent to create it.
    /// </summary>
    public abstract class SensorComponent : MonoBehaviour
    {
        /// <summary>
        /// Create the ISensor. This is called by the Agent when it is initialized.
        /// </summary>
        /// <returns></returns>
        public abstract ISensor CreateSensor();

        /// <summary>
        /// Returns the shape of the sensor observations that will be created.
        /// </summary>
        /// <returns></returns>
        public abstract int[] GetObservationShape();

        public virtual bool IsVisual()
        {
            var shape = GetObservationShape();
            return shape.Length == 3;
        }

        public virtual bool IsVector()
        {
            var shape = GetObservationShape();
            return shape.Length == 1;
        }
    }
}
