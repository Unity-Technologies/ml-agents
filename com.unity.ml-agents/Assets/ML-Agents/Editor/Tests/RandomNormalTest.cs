using System;
using NUnit.Framework;
using MLAgents.InferenceBrain.Utils;

namespace MLAgents.Tests
{
    public class RandomNormalTest
    {
        const float k_FirstValue = -1.19580f;
        const float k_SecondValue = -0.97345f;
        const double k_Epsilon = 0.0001;

        [Test]
        public void RandomNormalTestTwoDouble()
        {
            var rn = new RandomNormal(2018);

            Assert.AreEqual(k_FirstValue, rn.NextDouble(), k_Epsilon);
            Assert.AreEqual(k_SecondValue, rn.NextDouble(), k_Epsilon);
        }

        [Test]
        public void RandomNormalTestWithMean()
        {
            var rn = new RandomNormal(2018, 5.0f);

            Assert.AreEqual(k_FirstValue + 5.0, rn.NextDouble(), k_Epsilon);
            Assert.AreEqual(k_SecondValue + 5.0, rn.NextDouble(), k_Epsilon);
        }

        [Test]
        public void RandomNormalTestWithStddev()
        {
            var rn = new RandomNormal(2018, 0.0f, 4.2f);

            Assert.AreEqual(k_FirstValue * 4.2, rn.NextDouble(), k_Epsilon);
            Assert.AreEqual(k_SecondValue * 4.2, rn.NextDouble(), k_Epsilon);
        }

        [Test]
        public void RandomNormalTestWithMeanStddev()
        {
            const float mean = -3.2f;
            const float stddev = 2.2f;
            var rn = new RandomNormal(2018, mean, stddev);

            Assert.AreEqual(k_FirstValue * stddev + mean, rn.NextDouble(), k_Epsilon);
            Assert.AreEqual(k_SecondValue * stddev + mean, rn.NextDouble(), k_Epsilon);
        }

        [Test]
        public void RandomNormalTestDistribution()
        {
            const float mean = -3.2f;
            const float stddev = 2.2f;
            var rn = new RandomNormal(2018, mean, stddev);

            const int numSamples = 100000;
            // Adapted from https://www.johndcook.com/blog/standard_deviation/
            // Computes stddev and mean without losing precision
            double oldM = 0.0, newM = 0.0, oldS = 0.0, newS = 0.0;

            for (var i = 0; i < numSamples; i++)
            {
                var x = rn.NextDouble();
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

            var sampleMean = newM;
            var sampleVariance = newS / (numSamples - 1);
            var sampleStddev = Math.Sqrt(sampleVariance);

            // Note a larger epsilon here. We could get closer to the true values with more samples.
            Assert.AreEqual(mean, sampleMean, 0.01);
            Assert.AreEqual(stddev, sampleStddev, 0.01);
        }
    }
}
