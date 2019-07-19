using System;
using NUnit.Framework;
using MLAgents.InferenceBrain;
using MLAgents.InferenceBrain.Utils;

using UnityEngine;
using System.Collections;



namespace MLAgents.Tests
{

    public class RandomNormalTest
    {
        private const float first = -1.19580f;
        private const float second = -0.97345f;

        [Test]
        public void RandomNormalTestTwoDouble ()
        {
            RandomNormal rn = new RandomNormal (2018);

            Assert.AreEqual (first, rn.NextDouble (), 0.0001);
            Assert.AreEqual (second, rn.NextDouble (), 0.0001);
        }

        [Test]
        public void RandomNormalTestWithMean ()
        {
            RandomNormal rn = new RandomNormal (2018, 5.0f);

            Assert.AreEqual (first + 5.0, rn.NextDouble (), 0.0001);
            Assert.AreEqual (second + 5.0, rn.NextDouble (), 0.0001);
        }

        [Test]
        public void RandomNormalTestWithStddev ()
        {
            RandomNormal rn = new RandomNormal (2018, 0.0f, 4.2f);

            Assert.AreEqual (first * 4.2, rn.NextDouble (), 0.0001);
            Assert.AreEqual (second * 4.2, rn.NextDouble (), 0.0001);
        }

        [Test]
        public void RandomNormalTestWithMeanStddev ()
        {
            float mean = -3.2f;
            float stddev = 2.2f;
            RandomNormal rn = new RandomNormal (2018, mean, stddev);

            Assert.AreEqual (first * stddev + mean, rn.NextDouble (), 0.0001);
            Assert.AreEqual (second * stddev + mean, rn.NextDouble (), 0.0001);
        }

        [Test]
        public void RandomNormalTestTensorInt ()
        {
            RandomNormal rn = new RandomNormal (1982);
            Tensor t = new Tensor {
                ValueType = Tensor.TensorType.Integer
            };

            Assert.Throws<NotImplementedException> (() => rn.FillTensor (t));
        }

        [Test]
        public void RandomNormalTestDataNull ()
        {
            RandomNormal rn = new RandomNormal (1982);
            Tensor t = new Tensor {
                ValueType = Tensor.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException> (() => rn.FillTensor (t));
        }

        [Test]
        public void RandomNormalTestDistribution ()
        {
            float mean = -3.2f;
            float stddev = 2.2f;
            RandomNormal rn = new RandomNormal (2018, mean, stddev);

            int numSamples = 100000;
            // Adapted from https://www.johndcook.com/blog/standard_deviation/
            // Computes stddev and mean without losing precision
            double m_oldM = 0.0, m_newM = 0.0, m_oldS = 0.0, m_newS = 0.0;

            for (int i = 0; i < numSamples; i++) {
                double x = rn.NextDouble ();
                if (i == 0) {
                    m_oldM = m_newM = x;
                    m_oldS = 0.0;
                } else {
                    m_newM = m_oldM + (x - m_oldM) / i;
                    m_newS = m_oldS + (x - m_oldM) * (x - m_newM);

                    // set up for next iteration
                    m_oldM = m_newM;
                    m_oldS = m_newS;
                }
            }

            double sampleMean = m_newM;
            double sampleVariance = m_newS / (numSamples - 1);
            double sampleStddev = Math.Sqrt (sampleVariance);

            Assert.AreEqual (mean, sampleMean, 0.01);
            Assert.AreEqual (stddev, sampleStddev, 0.01);

        }

        [Test]
        public void RandomNormalTestTensor ()
        {
            RandomNormal rn = new RandomNormal (1982);
            Tensor t = new Tensor {
                ValueType = Tensor.TensorType.FloatingPoint,
                Data = Array.CreateInstance (typeof (float), new long [3] { 3, 4, 2 })
            };

            rn.FillTensor (t);

            float [] reference = new float []
            {
                -0.4315872f,
                0.9561074f,
                -1.130287f,
                -0.7763879f,
                -0.3027347f,
                -0.1377991f,
                -0.02921959f,
                0.9520947f,
                -1.11074f,
                -0.5018106f,
                0.1413168f,
                -0.07491868f,
                -0.2645015f,
                0.3331701f,
                0.3716498f,
                1.088157f,
                0.3414804f,
                1.167787f,
                -0.5105762f,
                0.5396146f,
                1.225356f,
                0.06144788f,
                -1.092338f,
                -1.177194f,
            };

            int i = 0;
            foreach (float f in t.Data) {
                Assert.AreEqual (f, reference [i], 0.0001);
                ++i;
            }


        }
    }
}
