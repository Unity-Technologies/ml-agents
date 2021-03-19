using NUnit.Framework;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    public class Float2DSensor : ISensor
    {
        public int Width { get; }
        public int Height { get; }
        string m_Name;
        private ObservationSpec m_ObservationSpec;
        public float[,] floatData;

        public Float2DSensor(int width, int height, string name)
        {
            Width = width;
            Height = height;
            m_Name = name;

            m_ObservationSpec = ObservationSpec.Visual(height, width, 1);
            floatData = new float[Height, Width];
        }

        public Float2DSensor(float[,] floatData, string name)
        {
            this.floatData = floatData;
            Height = floatData.GetLength(0);
            Width = floatData.GetLength(1);
            m_Name = name;
            m_ObservationSpec = ObservationSpec.Visual(Height, Width, 1);
        }

        public string GetName()
        {
            return m_Name;
        }

        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        public byte[] GetCompressedObservation()
        {
            return null;
        }

        public int Write(ObservationWriter writer)
        {
            using (TimerStack.Instance.Scoped("Float2DSensor.Write"))
            {
                for (var h = 0; h < Height; h++)
                {
                    for (var w = 0; w < Width; w++)
                    {
                        writer[h, w, 0] = floatData[h, w];
                    }
                }
                var numWritten = Height * Width;
                return numWritten;
            }
        }

        public void Update() { }
        public void Reset() { }

        public CompressionSpec GetCompressionSpec()
        {
            return CompressionSpec.Default();
        }
    }

    public class FloatVisualSensorTests
    {
        [Test]
        public void TestFloat2DSensorWrite()
        {
            var sensor = new Float2DSensor(3, 4, "floatsensor");
            for (var h = 0; h < 4; h++)
            {
                for (var w = 0; w < 3; w++)
                {
                    sensor.floatData[h, w] = 3 * h + w;
                }
            }

            var output = new float[12];
            var writer = new ObservationWriter();
            writer.SetTarget(output, sensor.GetObservationSpec(), 0);
            sensor.Write(writer);
            for (var i = 0; i < 9; i++)
            {
                Assert.AreEqual(i, output[i]);
            }
        }

        [Test]
        public void TestFloat2DSensorExternalData()
        {
            var data = new float[4, 3];
            var sensor = new Float2DSensor(data, "floatsensor");
            Assert.AreEqual(sensor.Height, 4);
            Assert.AreEqual(sensor.Width, 3);
        }
    }
}
