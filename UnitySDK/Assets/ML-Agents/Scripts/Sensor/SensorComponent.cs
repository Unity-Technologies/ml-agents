using System;
using UnityEngine;

namespace MLAgents.Sensor
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
    }
}
