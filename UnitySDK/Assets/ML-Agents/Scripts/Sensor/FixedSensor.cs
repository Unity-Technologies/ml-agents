namespace MLAgents.Sensor
{
    public class FixedSensor : ISensor
    {
        Observation m_Observation;
        string m_Name;

        public FixedSensor(string name, Observation observation)
        {
            m_Name = name;
            m_Observation = observation;
        }

        public string GetName()
        {
            return m_Name;
        }

        public int[] GetObservationShape()
        {
            return m_Observation.Shape;
        }

        public byte[] GetCompressedObservation()
        {
            return m_Observation.CompressedData;
        }

        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("NullSensor.WriteToTensor"))
            {
                for (int i = 0; i < m_Observation.FloatData.Count; i++)
                {
                    adapter[i] = m_Observation.FloatData.Array[i];
                }
                return m_Observation.FloatData.Count;
            }
        }

        public void Update() { }

        public SensorCompressionType GetCompressionType()
        {
            return m_Observation.CompressionType;
        }
    }
}
