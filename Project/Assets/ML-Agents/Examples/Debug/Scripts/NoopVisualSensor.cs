using MLAgents;

namespace MLAgentsExamples
{
    public class NoopVisualSensor : ISensor
    {
        int Width { get; }
        int Height { get; }
        string m_Name;
        int[] m_Shape;
        bool Grayscale;

        public NoopVisualSensor(int width, int height, bool grayscale, string name)
        {
            Width = width;
            Height = height;
            Grayscale = grayscale;
            m_Name = name;
            m_Shape = new[] { height, width, Grayscale ? 1 : 3 };
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
            using (TimerStack.Instance.Scoped("NoopVisualSensor.Write"))
            {
                var numChannels = (Grayscale ? 1 : 3);
                for (var h = 0; h < Height; h++)
                {
                    for (var w = 0; w < Width; w++)
                    {
                        for (var c = 0; c < numChannels; c++)
                        {
                            adapter[h, w, c] = 0.5f;
                        }
                    }
                }
                var numWritten = Height * Width * numChannels;
                return numWritten;
            }
        }

        public void Update() {}

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }
    }
}
