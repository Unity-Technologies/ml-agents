using System;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// A Sensor that allows to observe a variable number of entities.
    /// </summary>
    public class BufferSensor : ISensor, IBuiltInSensor
    {
        private string m_Name;
        private int m_MaxNumObs;
        private int m_ObsSize;
        float[] m_ObservationBuffer;
        int m_CurrentNumObservables;
        ObservationSpec m_ObservationSpec;


        /// <summary>
        /// Creates the BufferSensor.
        /// </summary>
        /// <param name="maxNumberObs">The maximum number of observations to be appended to this BufferSensor.</param>
        /// <param name="obsSize">The size of each observation appended to the BufferSensor.</param>
        /// <param name="name">The name of the sensor.</param>
        public BufferSensor(int maxNumberObs, int obsSize, string name)
        {
            m_Name = name;
            m_MaxNumObs = maxNumberObs;
            m_ObsSize = obsSize;
            m_ObservationBuffer = new float[m_ObsSize * m_MaxNumObs];
            m_CurrentNumObservables = 0;
            m_ObservationSpec = ObservationSpec.VariableLength(m_MaxNumObs, m_ObsSize);
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
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
            if (obs.Length != m_ObsSize)
            {
                throw new UnityAgentsException(
                    "The BufferSensor was expecting an observation of size " +
                    $"{m_ObsSize} but received {obs.Length} observations instead."
                );
            }
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
            // for (int i = 0; i < m_ObsSize * m_MaxNumObs; i++)
            // {
            //     writer[i] = m_ObservationBuffer[i];
            // }

            for (int i = 0; i < m_MaxNumObs; i++)
            {
                for (int j = 0; j < m_ObsSize; j++)
                {
                    writer[i, j] = m_ObservationBuffer[i * m_ObsSize + j];
                }
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

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            return CompressionSpec.Default();
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_Name;
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            return BuiltInSensorType.BufferSensor;
        }
    }
}
