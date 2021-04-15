using System;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Unity.MLAgents.Integrations.Match3
{
    /// <summary>
    /// Sensor component for a Match3 game.
    /// </summary>
    [AddComponentMenu("ML Agents/Match 3 Sensor", (int)MenuGroup.Sensors)]
    public class Match3SensorComponent : SensorComponent, IDisposable
    {
        /// <summary>
        /// Name of the generated Match3Sensor object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName = "Match3 Sensor";

        /// <summary>
        /// Type of observation to generate.
        /// </summary>
        public Match3ObservationType ObservationType = Match3ObservationType.Vector;

        private ISensor[] m_Sensors;

        /// <inheritdoc/>
        public override ISensor[] CreateSensors()
        {
            // Clean up any existing sensors
            Dispose();

            var board = GetComponent<AbstractBoard>();
            var cellSensor = Match3Sensor.CellTypeSensor(board, ObservationType, SensorName + " (cells)");
            // This can be null if numSpecialTypes is 0
            var specialSensor = Match3Sensor.SpecialTypeSensor(board, ObservationType, SensorName + " (special)");
            m_Sensors = specialSensor != null
                ? new ISensor[] { cellSensor, specialSensor }
            : new ISensor[] { cellSensor };
            return m_Sensors;
        }

        /// <summary>
        /// Clean up the sensors created by CreateSensors().
        /// </summary>
        public void Dispose()
        {
            if (m_Sensors != null)
            {
                for (var i = 0; i < m_Sensors.Length; i++)
                {
                    ((Match3Sensor)m_Sensors[i]).Dispose();
                }

                m_Sensors = null;
            }
        }
    }
}
