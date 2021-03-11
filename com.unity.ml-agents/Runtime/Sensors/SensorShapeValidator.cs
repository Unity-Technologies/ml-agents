using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    internal class SensorShapeValidator
    {
        List<ObservationSpec> m_SensorShapes;

        /// <summary>
        /// Check that the List Sensors are the same shape as the previous ones.
        /// If this is the first List of Sensors being checked, its Sensor sizes will be saved.
        /// </summary>
        public void ValidateSensors(List<ISensor> sensors)
        {
            if (m_SensorShapes == null)
            {
                m_SensorShapes = new List<ObservationSpec>(sensors.Count);
                // First agent, save the sensor sizes
                foreach (var sensor in sensors)
                {
                    m_SensorShapes.Add(sensor.GetObservationSpec());
                }
            }
            else
            {
                // Check for compatibility with the other Agents' Sensors
                // TODO make sure this only checks once per agent
                Debug.AssertFormat(
                    m_SensorShapes.Count == sensors.Count,
                    "Number of Sensors must match. {0} != {1}",
                    m_SensorShapes.Count,
                    sensors.Count
                );
                for (var i = 0; i < Mathf.Min(m_SensorShapes.Count, sensors.Count); i++)
                {
                    var cachedSpec = m_SensorShapes[i];
                    var sensorSpec = sensors[i].GetObservationSpec();
                    Debug.Assert(cachedSpec.Shape.Length == sensorSpec.Shape.Length, "Sensor dimensions must match.");
                    for (var j = 0; j < Mathf.Min(cachedSpec.Shape.Length, sensorSpec.Shape.Length); j++)
                    {
                        Debug.Assert(cachedSpec.Shape[j] == sensorSpec.Shape[j], "Sensor sizes must match.");
                    }
                }
            }
        }
    }
}
