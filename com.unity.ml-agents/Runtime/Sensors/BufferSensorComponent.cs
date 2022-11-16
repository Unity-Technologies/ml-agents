using UnityEngine;

namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// A SensorComponent that creates a <see cref="BufferSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Buffer Sensor", (int)MenuGroup.Sensors)]
    public class BufferSensorComponent : SensorComponent
    {

        /// <summary>
        /// Name of the generated <see cref="BufferSensor"/> object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get { return m_SensorName; }
            set { m_SensorName = value; }
        }
        [HideInInspector, SerializeField]
        private string m_SensorName = "BufferSensor";

        /// <summary>
        /// This is how many floats each entities will be represented with. This number
        /// is fixed and all entities must have the same representation.
        /// </summary>
        public int ObservableSize
        {
            get { return m_ObservableSize; }
            set { m_ObservableSize = value; }
        }
        [HideInInspector, SerializeField]
        private int m_ObservableSize;

        /// <summary>
        /// This is the maximum number of entities the `BufferSensor` will be able to
        /// collect.
        /// </summary>
        public int MaxNumObservables
        {
            get { return m_MaxNumObservables; }
            set { m_MaxNumObservables = value; }
        }
        [HideInInspector, SerializeField]
        private int m_MaxNumObservables;

        private BufferSensor m_Sensor;

        /// <inheritdoc/>
        public override ISensor[] CreateSensors()
        {
            m_Sensor = new BufferSensor(MaxNumObservables, ObservableSize, m_SensorName);
            return new ISensor[] { m_Sensor };
        }

        /// <summary>
        /// Appends an observation to the buffer. If the buffer is full (maximum number
        /// of observation is reached) the observation will be ignored. the length of
        /// the provided observation array must be equal to the observation size of
        /// the buffer sensor.
        /// </summary>
        /// <param name="obs"> The float array observation</param>
        public void AppendObservation(float[] obs)
        {
            m_Sensor.AppendObservation(obs);
        }
    }
}
