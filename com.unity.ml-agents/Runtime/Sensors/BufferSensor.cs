using System;

namespace Unity.MLAgents.Sensors
{
    internal class BufferSensor : ISensor, IDimensionPropertiesSensor, IBuiltInSensor
    {
        private int m_MaxNumObs;
        private int m_ObsSize;
        float[] m_ObservationBuffer;
        int m_CurrentNumObservables;
        public BufferSensor(int maxNumberObs, int obsSize)
        {
            m_MaxNumObs = maxNumberObs;
            m_ObsSize = obsSize;
            m_ObservationBuffer = new float[m_ObsSize * m_MaxNumObs];
            m_CurrentNumObservables = 0;
        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return new int[] { m_MaxNumObs, m_ObsSize };
        }

        /// <inheritdoc/>
        public DimensionProperty[] GetDimensionProperties()
        {
            return new DimensionProperty[]{
                DimensionProperty.VariableSize,
                DimensionProperty.None
            };
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
            if (m_CurrentNumObservables >= m_MaxNumObs)
            {
                return;
            }
            for (int i = 0; i < obs.Length; i++)
            {
                m_ObservationBuffer[m_CurrentNumObservables * m_ObsSize + i] = obs[i];
            }
            m_CurrentNumObservables++;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            for (int i = 0; i < m_ObsSize * m_MaxNumObs; i++)
            {
                writer[i] = m_ObservationBuffer[i];
            }
            return m_ObsSize * m_MaxNumObs;
        }

        /// <inheritdoc/>
        public virtual byte[] GetCompressedObservation()
        {
            return null;
        }

        /// <inheritdoc/>
        public void Update()
        {
            Reset();
        }

        /// <inheritdoc/>
        public void Reset()
        {
            m_CurrentNumObservables = 0;
            Array.Clear(m_ObservationBuffer, 0, m_ObservationBuffer.Length);
        }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        public string GetName()
        {
            return "BufferSensor";
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            return BuiltInSensorType.BufferSensor;
        }

    }

}
