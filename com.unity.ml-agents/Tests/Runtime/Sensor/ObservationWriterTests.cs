using NUnit.Framework;
using Unity.Barracuda;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Inference;


namespace Unity.MLAgents.Tests
{
    public class ObservationWriterTests
    {
        [Test]
        public void TestWritesToIList()
        {
            ObservationWriter writer = new ObservationWriter();
            var buffer = new[] { 0f, 0f, 0f };
            var shape = new InplaceArray<int>(3);

            writer.SetTarget(buffer, shape, 0);
            // Elementwise writes
            writer[0] = 1f;
            writer[2] = 2f;
            Assert.AreEqual(new[] { 1f, 0f, 2f }, buffer);

            // Elementwise writes with offset
            writer.SetTarget(buffer, shape, 1);
            writer[0] = 3f;
            Assert.AreEqual(new[] { 1f, 3f, 2f }, buffer);

            // AddList
            writer.SetTarget(buffer, shape, 0);
            writer.AddList(new[] { 4f, 5f });
            Assert.AreEqual(new[] { 4f, 5f, 2f }, buffer);

            // AddList with offset
            writer.SetTarget(buffer, shape, 1);
            writer.AddList(new[] { 6f, 7f });
            Assert.AreEqual(new[] { 4f, 6f, 7f }, buffer);
        }

        [Test]
        public void TestWritesToTensor()
        {
            ObservationWriter writer = new ObservationWriter();
            var srcData = new float[6];
            var t = new TensorProxy
            {
                valueType = DataType.Float,
                data = new TensorFloat(new TensorShape(2, 3), srcData)
            };

            writer.SetTarget(t, 0, 0);
            Assert.AreEqual(0f, t.FloatData[0, 0]);
            writer[0] = 1f;
            Assert.AreEqual(1f, t.FloatData[0, 0]);

            writer.SetTarget(t, 1, 1);
            writer[0] = 2f;
            writer[1] = 3f;
            // [0, 0] shouldn't change
            Assert.AreEqual(1f, t.FloatData[0, 0]);
            Assert.AreEqual(2f, t.FloatData[1, 1]);
            Assert.AreEqual(3f, t.FloatData[1, 2]);

            // AddList
            var srcData2 = new float[6];
            t = new TensorProxy
            {
                valueType = DataType.Float,
                data = new TensorFloat(new TensorShape(2, 3), srcData2)
            };

            writer.SetTarget(t, 1, 1);
            writer.AddList(new[] { -1f, -2f });
            Assert.AreEqual(0f, t.FloatData[0, 0]);
            Assert.AreEqual(0f, t.FloatData[0, 1]);
            Assert.AreEqual(0f, t.FloatData[0, 2]);
            Assert.AreEqual(0f, t.FloatData[1, 0]);
            Assert.AreEqual(-1f, t.FloatData[1, 1]);
            Assert.AreEqual(-2f, t.FloatData[1, 2]);
        }

        [Test]
        public void TestWritesToTensor3D()
        {
            ObservationWriter writer = new ObservationWriter();
            var srcData = new float[24];
            var t = new TensorProxy
            {
                valueType = DataType.Float,
                data = new TensorFloat(new TensorShape(2, 2, 2, 3), srcData)
            };

            writer.SetTarget(t, 0, 0);
            writer[1, 1, 0] = 1f;
            Assert.AreEqual(1f, t.FloatData[0, 0, 1, 1]);

            writer.SetTarget(t, 0, 1);
            writer[1, 0, 0] = 2f;
            Assert.AreEqual(2f, t.FloatData[0, 0, 1, 1]);
        }
    }
}
