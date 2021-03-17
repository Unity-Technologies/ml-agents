using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Sensors
{
    [AddComponentMenu("ML Agents/Vector Sensor", (int)MenuGroup.Sensors)]
    public class VectorSensorComponent : SensorComponent
    {
        public int ObservationSize
        {
            get { return m_observationSize; }
            set { m_observationSize = value; }
        }

        [HideInInspector, SerializeField]
        int m_observationSize;

        [HideInInspector, SerializeField]
        ObservationType m_ObservationType;

        VectorSensor m_sensor;

        public ObservationType ObservationType
        {
            get { return m_ObservationType; }
            set { m_ObservationType = value; }
        }

        /// <summary>
        /// Creates a VectorSensor.
        /// </summary>
        /// <returns></returns>
        public override ISensor CreateSensor()
        {
            m_sensor = new VectorSensor(m_observationSize, observationType: m_ObservationType);
            return m_sensor;
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            return new[] { m_observationSize };
        }

        public VectorSensor GetSensor()
        {
            return m_sensor;
        }


        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            if (m_sensor != null)
            {

            }
        }
    }
}
