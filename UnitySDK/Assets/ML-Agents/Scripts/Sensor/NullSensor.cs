namespace MLAgents.Sensor
{
    public class NullSensor : ISensor
    {
        int[] m_Shape;
        string m_Name;

        public NullSensor(string name, int[] shape)
        {
            m_Name = name;
            m_Shape = shape;
        }

        public string GetName()
        {
            return m_Name;
        }

        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        public byte[] GetCompressedObservation()
        {
            return null;
        }

        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("NullSensor.WriteToTensor"))
            {
                int numWritten = 0;
                if (m_Shape.Length == 1)
                {
                    for (int i = 0; i < m_Shape[0]; i++)
                    {
                        adapter[i] = 0f;
                        numWritten++;
                    }
                }
                else if (m_Shape.Length == 3)
                {
                    for (int u = 0; u < m_Shape[0]; u++)
                    {
                        for (int v = 0; v < m_Shape[1]; v++)
                        {
                            for (int w = 0; w < m_Shape[2]; w++)
                            {
                                adapter[u, v, w] = 0f;
                                numWritten++;
                            }
                        }
                    }
                }
                return numWritten;
            }
        }

        public void Update() { }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }
    }
}
