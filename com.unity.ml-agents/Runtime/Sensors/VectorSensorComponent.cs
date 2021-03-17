using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Sensors
{
    [AddComponentMenu("ML Agents/Vector Sensor", (int)MenuGroup.Sensors)]
    public class VectorSensorComponent : SensorComponent
    {

        /// <summary>
        /// Name of the generated <see cref="VectorSensor"/> object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get { return m_SensorName; }
            set { m_SensorName = value; }
        }
        [HideInInspector, SerializeField]
        private string m_SensorName = "VectorSensor";

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
            m_sensor = new VectorSensor(m_observationSize, m_ObservationType, m_SensorName);
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
    }
}
