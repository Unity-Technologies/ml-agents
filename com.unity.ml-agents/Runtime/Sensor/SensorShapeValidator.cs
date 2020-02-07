using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    internal class SensorShapeValidator
    {
        List<int[]> m_SensorShapes;

        /// <summary>
        /// Check that the List Sensors are the same shape as the previous ones.
        /// If this is the first List of Sensors being checked, its Sensor sizes will be saved.
        /// </summary>
        public void ValidateSensors(List<ISensor> sensors)
        {
            if (m_SensorShapes == null)
            {
                m_SensorShapes = new List<int[]>(sensors.Count);
                // First agent, save the sensor sizes
                foreach (var sensor in sensors)
                {
                    m_SensorShapes.Add(sensor.GetObservationShape());
                }
            }
            else
            {
                // Check for compatibility with the other Agents' Sensors
                // TODO make sure this only checks once per agent
                Debug.Assert(m_SensorShapes.Count == sensors.Count, $"Number of Sensors must match. {m_SensorShapes.Count} != {sensors.Count}");
                for (var i = 0; i < m_SensorShapes.Count; i++)
                {
                    var cachedShape = m_SensorShapes[i];
                    var sensorShape = sensors[i].GetObservationShape();
                    Debug.Assert(cachedShape.Length == sensorShape.Length, "Sensor dimensions must match.");
                    for (var j = 0; j < cachedShape.Length; j++)
                    {
                        Debug.Assert(cachedShape[j] == sensorShape[j], "Sensor sizes much match.");
                    }
                }
            }
        }
    }
}
