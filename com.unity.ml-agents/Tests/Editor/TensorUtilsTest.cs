using System;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Inference.Utils;
using Unity.Barracuda;
using NUnit.Framework;

namespace Unity.MLAgents.Tests
{
    public class TensorUtilsTest
    {
        [TestCase(4, TestName = "TestResizeTensor_4D")]
        [TestCase(8, TestName = "TestResizeTensor_8D")]
        public void TestResizeTensor(int dimension)
        {
            if (dimension == 8)
            {
                // Barracuda 1.0.x doesn't support 8D tensors
                // Barracuda 1.1.x does but it initially broke ML-Agents support
                // Unfortunately, the PackageInfo methods don't exist in earlier versions of the editor,
                // so just skip that variant of the test then.
                // It's unlikely, but possible that we'll upgrade to a newer dependency of Barracuda,
                // in which case we should make sure this test is run then.
#if UNITY_2019_3_OR_NEWER
                var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(Tensor).Assembly);
                Assert.AreEqual("com.unity.barracuda", packageInfo.name);
                var barracuda8DSupport = new Version(1, 1, 0);
                var strippedBarracudaVersion = packageInfo.version.Replace("-preview", "");
                var version = new Version(strippedBarracudaVersion);
                if (version <= barracuda8DSupport)
                {
                    return;
                }
#else
                return;
#endif
            }
            var alloc = new TensorCachingAllocator();
            var height = 64;
            var width = 84;
            var channels = 3;

            // Set shape to {1, ..., height, width, channels}
            // For 8D, the ... are all 1's
            var shape = new long[dimension];
            for (var i = 0; i < dimension; i++)
            {
                shape[i] = 1;
            }

            shape[dimension - 3] = height;
            shape[dimension - 2] = width;
            shape[dimension - 1] = channels;

            var intShape = new int[dimension];
            for (var i = 0; i < dimension; i++)
            {
                intShape[i] = (int)shape[i];
            }

            var tensorProxy = new TensorProxy
            {
                valueType = TensorProxy.TensorType.Integer,
                data = new Tensor(intShape),
                shape = shape,
            };

            // These should be invariant after the resize.
            Assert.AreEqual(height, tensorProxy.data.shape.height);
            Assert.AreEqual(width, tensorProxy.data.shape.width);
            Assert.AreEqual(channels, tensorProxy.data.shape.channels);

            TensorUtils.ResizeTensor(tensorProxy, 42, alloc);

            Assert.AreEqual(height, tensorProxy.shape[dimension - 3]);
            Assert.AreEqual(width, tensorProxy.shape[dimension - 2]);
            Assert.AreEqual(channels, tensorProxy.shape[dimension - 1]);

            Assert.AreEqual(height, tensorProxy.data.shape.height);
            Assert.AreEqual(width, tensorProxy.data.shape.width);
            Assert.AreEqual(channels, tensorProxy.data.shape.channels);

            alloc.Dispose();
        }

        [Test]
        public void RandomNormalTestTensorInt()
        {
            var rn = new RandomNormal(1982);
            var t = new TensorProxy
            {
                valueType = TensorProxy.TensorType.Integer
            };

            Assert.Throws<NotImplementedException>(
                () => TensorUtils.FillTensorWithRandomNormal(t, rn));
        }

        [Test]
        public void RandomNormalTestDataNull()
        {
            var rn = new RandomNormal(1982);
            var t = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(
                () => TensorUtils.FillTensorWithRandomNormal(t, rn));
        }

        [Test]
        public void RandomNormalTestTensor()
        {
            var rn = new RandomNormal(1982);
            var t = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor(1, 3, 4, 2)
            };

            TensorUtils.FillTensorWithRandomNormal(t, rn);

            var reference = new[]
            {
                -0.4315872f,
                -1.11074f,
                0.3414804f,
                -1.130287f,
                0.1413168f,
                -0.5105762f,
                -0.3027347f,
                -0.2645015f,
                1.225356f,
                -0.02921959f,
                0.3716498f,
                -1.092338f,
                0.9561074f,
                -0.5018106f,
                1.167787f,
                -0.7763879f,
                -0.07491868f,
                0.5396146f,
                -0.1377991f,
                0.3331701f,
                0.06144788f,
                0.9520947f,
                1.088157f,
                -1.177194f,
            };

            for (var i = 0; i < t.data.length; i++)
            {
                Assert.AreEqual(t.data[i], reference[i], 0.0001);
            }
        }
    }
}
