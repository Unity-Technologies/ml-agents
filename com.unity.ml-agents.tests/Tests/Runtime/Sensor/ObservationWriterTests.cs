using NUnit.Framework;
using Unity.InferenceEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Inference;
using UnityEngine;


namespace Unity.MLAgents.Tests
{
    public class ObservationWriterTests
    {
        static TensorProxy MakeTensorProxy(int batch, int width)
        {
            return new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor<float>(new TensorShape(batch, width))
            };
        }
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
            var t = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor<float>(new TensorShape(2, 3))
            };

            writer.SetTarget(t, 0, 0);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 0]);
            writer[0] = 1f;
            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0]);

            writer.SetTarget(t, 1, 1);
            writer[0] = 2f;
            writer[1] = 3f;
            // [0, 0] shouldn't change
            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)t.data)[1, 1]);
            Assert.AreEqual(3f, ((Tensor<float>)t.data)[1, 2]);
            t.data.Dispose();

            // AddList
            t = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor<float>(new TensorShape(2, 3))
            };

            writer.SetTarget(t, 1, 1);
            writer.AddList(new[] { -1f, -2f });
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 1]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 2]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[1, 0]);
            Assert.AreEqual(-1f, ((Tensor<float>)t.data)[1, 1]);
            Assert.AreEqual(-2f, ((Tensor<float>)t.data)[1, 2]);
            t.data.Dispose();
        }

        [Test]
        public void TestWritesToTensor3D()
        {
            ObservationWriter writer = new ObservationWriter();
            var t = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor<float>(new TensorShape(2, 3, 2, 2))
            };

            writer.SetTarget(t, 0, 0);
            writer[1, 1, 0] = 1f;
            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 1, 1, 0]);

            writer.SetTarget(t, 0, 1);
            writer[0, 1, 0] = 2f;
            Assert.AreEqual(2f, ((Tensor<float>)t.data)[0, 1, 1, 0]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_1DIndexer()
        {
            var writer = new ObservationWriter();
            var t = MakeTensorProxy(1, 3);

            writer.SetTarget(t, 0, 0);
            writer[0] = 1f;
            writer[1] = 2f;
            writer[2] = 3f;
            // Index 3 is out of bounds — must be silently dropped, not crash
            writer[3] = 99f;
            writer[100] = 99f;

            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)t.data)[0, 1]);
            Assert.AreEqual(3f, ((Tensor<float>)t.data)[0, 2]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_1DIndexerWithOffset()
        {
            var writer = new ObservationWriter();
            var t = MakeTensorProxy(1, 4);

            writer.SetTarget(t, 0, 2);
            writer[0] = 10f;
            writer[1] = 11f;
            // offset(2) + index(2) = 4 => out of bounds
            writer[2] = 99f;

            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 1]);
            Assert.AreEqual(10f, ((Tensor<float>)t.data)[0, 2]);
            Assert.AreEqual(11f, ((Tensor<float>)t.data)[0, 3]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_AddList()
        {
            var writer = new ObservationWriter();
            var t = MakeTensorProxy(1, 3);

            writer.SetTarget(t, 0, 1);
            // List has 5 elements but only 2 slots remain (capacity 3, offset 1)
            writer.AddList(new[] { 10f, 20f, 30f, 40f, 50f });

            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(10f, ((Tensor<float>)t.data)[0, 1]);
            Assert.AreEqual(20f, ((Tensor<float>)t.data)[0, 2]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_AddListWithWriteOffset()
        {
            var writer = new ObservationWriter();
            var t = MakeTensorProxy(1, 4);

            writer.SetTarget(t, 0, 1);
            // offset=1, writeOffset=2 => start at 3, only 1 slot left
            writer.AddList(new[] { 10f, 20f, 30f }, 2);

            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 1]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 2]);
            Assert.AreEqual(10f, ((Tensor<float>)t.data)[0, 3]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_Vector3()
        {
            var writer = new ObservationWriter();
            // Only 2 slots — Vector3 needs 3
            var t = MakeTensorProxy(1, 2);

            writer.SetTarget(t, 0, 0);
            writer.Add(new Vector3(1f, 2f, 3f));

            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)t.data)[0, 1]);
            // Third component silently dropped
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_Vector3AtCapacity()
        {
            var writer = new ObservationWriter();
            var t = MakeTensorProxy(1, 3);

            // offset=3 means 0 slots remain
            writer.SetTarget(t, 0, 3);
            writer.Add(new Vector3(1f, 2f, 3f));

            // All three slots should be untouched (zero)
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 1]);
            Assert.AreEqual(0f, ((Tensor<float>)t.data)[0, 2]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_Vector4()
        {
            var writer = new ObservationWriter();
            // Only 2 slots — Vector4 needs 4
            var t = MakeTensorProxy(1, 2);

            writer.SetTarget(t, 0, 0);
            writer.Add(new Vector4(1f, 2f, 3f, 4f));

            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)t.data)[0, 1]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_Quaternion()
        {
            var writer = new ObservationWriter();
            // Only 3 slots — Quaternion needs 4
            var t = MakeTensorProxy(1, 3);

            writer.SetTarget(t, 0, 0);
            writer.Add(new Quaternion(1f, 2f, 3f, 4f));

            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0]);
            Assert.AreEqual(2f, ((Tensor<float>)t.data)[0, 1]);
            Assert.AreEqual(3f, ((Tensor<float>)t.data)[0, 2]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_3DIndexer()
        {
            var writer = new ObservationWriter();
            // Shape: [1, 2, 3, 4] => 2 channels, 3 height, 4 width
            var t = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor<float>(new TensorShape(1, 2, 3, 4))
            };

            writer.SetTarget(t, 0, 0);

            // Valid write
            writer[0, 0, 0] = 1f;
            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0, 0, 0]);

            // Out-of-bounds channel — silently dropped
            writer[2, 0, 0] = 99f;
            // Out-of-bounds height — silently dropped
            writer[0, 3, 0] = 99f;
            // Out-of-bounds width — silently dropped
            writer[0, 0, 4] = 99f;
            // Negative height — silently dropped
            writer[0, -1, 0] = 99f;

            // Original value unchanged
            Assert.AreEqual(1f, ((Tensor<float>)t.data)[0, 0, 0, 0]);
            t.data.Dispose();
        }

        [Test]
        public void TestTensorBoundsClamp_InBoundsWritesStillWork()
        {
            var writer = new ObservationWriter();
            var t = MakeTensorProxy(2, 5);

            // Verify normal in-bounds writes are unaffected by the guards
            writer.SetTarget(t, 0, 0);
            writer.AddList(new[] { 1f, 2f, 3f, 4f, 5f });

            writer.SetTarget(t, 1, 0);
            writer.Add(new Vector3(10f, 20f, 30f));
            writer[3] = 40f;
            writer[4] = 50f;

            for (var i = 0; i < 5; i++)
                Assert.AreEqual(i + 1f, ((Tensor<float>)t.data)[0, i]);

            Assert.AreEqual(10f, ((Tensor<float>)t.data)[1, 0]);
            Assert.AreEqual(20f, ((Tensor<float>)t.data)[1, 1]);
            Assert.AreEqual(30f, ((Tensor<float>)t.data)[1, 2]);
            Assert.AreEqual(40f, ((Tensor<float>)t.data)[1, 3]);
            Assert.AreEqual(50f, ((Tensor<float>)t.data)[1, 4]);
            t.data.Dispose();
        }
    }
}
