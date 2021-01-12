using UnityEngine;

namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// A component for BufferSensor.
    /// </summary>
    [AddComponentMenu("ML Agents/Buffer Sensor", (int)MenuGroup.Sensors)]
    internal class BufferSensorComponent : SensorComponent
    {
        public int ObservableSize;
        public int MaxNumObservables;
        private BufferSensor m_Sensor;

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            m_Sensor = new BufferSensor(MaxNumObservables, ObservableSize);
            return m_Sensor;
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            return new[] { MaxNumObservables, ObservableSize };
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
