using System;
using Barracuda;
using MLAgents.InferenceBrain;
using MLAgents.InferenceBrain.Utils;
using NUnit.Framework;

namespace MLAgents.Tests
{
    public class TensorUtilsTest
    {
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
