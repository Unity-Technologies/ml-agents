using System;
using Barracuda;
using NUnit.Framework;
using MLAgents.InferenceBrain;
using MLAgents.InferenceBrain.Utils;


namespace MLAgents.Tests
{

    public class RandomNormalTest
    {

        private const float firstValue = -1.19580f;
        private const float secondValue = -0.97345f;
        private const double epsilon = 0.0001;

        [Test]
        public void RandomNormalTestTwoDouble()
        {
            RandomNormal rn = new RandomNormal(2018);

            Assert.AreEqual(firstValue, rn.NextDouble(), epsilon);
            Assert.AreEqual(secondValue, rn.NextDouble(), epsilon);
        }

        [Test]
        public void RandomNormalTestWithMean()
        {
            RandomNormal rn = new RandomNormal(2018, 5.0f);

            Assert.AreEqual(firstValue + 5.0, rn.NextDouble(), epsilon);
            Assert.AreEqual(secondValue + 5.0, rn.NextDouble(), epsilon);
        }

        [Test]
        public void RandomNormalTestWithStddev()
        {
            RandomNormal rn = new RandomNormal(2018, 0.0f, 4.2f);

            Assert.AreEqual(firstValue * 4.2, rn.NextDouble(), epsilon);
            Assert.AreEqual(secondValue * 4.2, rn.NextDouble(), epsilon);
        }

        [Test]
        public void RandomNormalTestWithMeanStddev()
        {
            float mean = -3.2f;
            float stddev = 2.2f;
            RandomNormal rn = new RandomNormal(2018, mean, stddev);

            Assert.AreEqual(firstValue * stddev + mean, rn.NextDouble(), epsilon);
            Assert.AreEqual(secondValue * stddev + mean, rn.NextDouble(), epsilon);
        }

        [Test]
        public void RandomNormalTestTensorInt()
        {
            RandomNormal rn = new RandomNormal(1982);
            TensorProxy t = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.Integer
            };

            Assert.Throws<NotImplementedException>(() => rn.FillTensor(t));
        }

        [Test]
        public void RandomNormalTestDataNull()
        {
            RandomNormal rn = new RandomNormal(1982);
            TensorProxy t = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(() => rn.FillTensor(t));
        }

        [Test]
        public void RandomNormalTestDistribution()
        {
            float mean = -3.2f;
            float stddev = 2.2f;
            RandomNormal rn = new RandomNormal(2018, mean, stddev);

            int numSamples = 100000;
            // Adapted from https://www.johndcook.com/blog/standard_deviation/
            // Computes stddev and mean without losing precision
            double oldM = 0.0, newM = 0.0, oldS = 0.0, newS = 0.0;

            for (int i = 0; i < numSamples; i++)
            {
                double x = rn.NextDouble();
                if (i == 0)
                {
                    oldM = newM = x;
                    oldS = 0.0;
                }
                else
                {
                    newM = oldM + (x - oldM) / i;
                    newS = oldS + (x - oldM) * (x - newM);

                    // set up for next iteration
                    oldM = newM;
                    oldS = newS;
                }
            }

            double sampleMean = newM;
            double sampleVariance = newS / (numSamples - 1);
            double sampleStddev = Math.Sqrt(sampleVariance);

            // Note a larger epsilon here. We could get closer to the true values with more samples.
            Assert.AreEqual(mean, sampleMean, 0.01);
            Assert.AreEqual(stddev, sampleStddev, 0.01);

        }

        [Test]
        public void RandomNormalTestTensor()
        {
            RandomNormal rn = new RandomNormal(1982);
            TensorProxy t = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint,
                Data = new Tensor(1, 3, 4, 2)
            };

            rn.FillTensor(t);

            float[] reference = new float[]
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

            for (var i = 0; i < t.Data.length; i++)
            {
                Assert.AreEqual(t.Data[i], reference[i], 0.0001);
            }
        }
    }
}
