using System;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Integrations.Match3
{
    /// <summary>
    /// Sensor component for a Match3 game.
    /// </summary>
    [AddComponentMenu("ML Agents/Match 3 Sensor", (int)MenuGroup.Sensors)]
    public class Match3SensorComponent : SensorComponent, IDisposable
    {
        [HideInInspector, SerializeField, FormerlySerializedAs("SensorName")]
        string m_SensorName = "Match3 Sensor";

        /// <summary>
        /// Name of the generated Match3Sensor object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get => m_SensorName;
            set => m_SensorName = value;
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("ObservationType")]
        Match3ObservationType m_ObservationType = Match3ObservationType.Vector;

        /// <summary>
        /// Type of observation to generate.
        /// </summary>
        public Match3ObservationType ObservationType
        {
            get => m_ObservationType;
            set => m_ObservationType = value;
        }

        private ISensor[] m_Sensors;

        /// <inheritdoc/>
        public override ISensor[] CreateSensors()
        {
            // Clean up any existing sensors
            Dispose();

            var board = GetComponent<AbstractBoard>();
            if (!board)
            {
                return Array.Empty<ISensor>();
            }
            var cellSensor = Match3Sensor.CellTypeSensor(board, m_ObservationType, m_SensorName + " (cells)");
            // This can be null if BoardSize.NumSpecialTypes is 0
            var specialSensor = Match3Sensor.SpecialTypeSensor(board, m_ObservationType, m_SensorName + " (special)");
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
