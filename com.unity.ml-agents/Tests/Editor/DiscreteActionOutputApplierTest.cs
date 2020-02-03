using System;
using Barracuda;
using NUnit.Framework;
using UnityEngine;
using MLAgents.InferenceBrain;
using MLAgents.InferenceBrain.Utils;

namespace MLAgents.Tests
{
    public class DiscreteActionOutputApplierTest
    {
        [Test]
        public void TestEvalP()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                data = new Tensor(1, 3, new[] {0.1f, 0.2f, 0.7f}),
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            var dst = new TensorProxy
            {
                data = new Tensor(1, 3),
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            DiscreteActionOutputApplier.Eval(src, dst, m);

            float[] reference = {2, 2, 1};
            for (var i = 0; i < dst.data.length; i++)
            {
                Assert.AreEqual(reference[i], dst.data[i]);
                ++i;
            }
        }

        [Test]
        public void TestEvalLogits()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                data = new Tensor(
                    1,
                    3,
                    new[] { Mathf.Log(0.1f) - 50, Mathf.Log(0.2f) - 50, Mathf.Log(0.7f) - 50 }),
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            var dst = new TensorProxy
            {
                data = new Tensor(1, 3),
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            DiscreteActionOutputApplier.Eval(src, dst, m);

            float[] reference = {2, 2, 2};
            for (var i = 0; i < dst.data.length; i++)
            {
                Assert.AreEqual(reference[i], dst.data[i]);
                ++i;
            }
        }

        [Test]
        public void TestEvalBatching()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                data = new Tensor(2, 3, new[]
                {
                    Mathf.Log(0.1f) - 50, Mathf.Log(0.2f) - 50, Mathf.Log(0.7f) - 50,
                    Mathf.Log(0.3f) - 25, Mathf.Log(0.4f) - 25, Mathf.Log(0.3f) - 25
                }),
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            var dst = new TensorProxy
            {
                data = new Tensor(2, 3),
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            DiscreteActionOutputApplier.Eval(src, dst, m);

            float[] reference = {2, 2, 2, 0, 1, 0};
            for (var i = 0; i < dst.data.length; i++)
            {
                Assert.AreEqual(reference[i], dst.data[i]);
                ++i;
            }
        }

        [Test]
        public void TestSrcInt()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                valueType = TensorProxy.TensorType.Integer
            };

            Assert.Throws<NotImplementedException>(
                () => DiscreteActionOutputApplier.Eval(src, null, m));
        }

        [Test]
        public void TestDstInt()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            var dst = new TensorProxy
            {
                valueType = TensorProxy.TensorType.Integer
            };

            Assert.Throws<ArgumentException>(
                () => DiscreteActionOutputApplier.Eval(src, dst, m));
        }

        [Test]
        public void TestSrcDataNull()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            var dst = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(
                () => DiscreteActionOutputApplier.Eval(src, dst, m));
        }

        [Test]
        public void TestDstDataNull()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor(0, 1)
            };

            var dst = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(
                () => DiscreteActionOutputApplier.Eval(src, dst, m));
        }

        [Test]
        public void TestUnequalBatchSize()
        {
            var m = new Multinomial(2018);

            var src = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor(1, 1)
            };

            var dst = new TensorProxy
            {
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = new Tensor(2, 1)
            };

            Assert.Throws<ArgumentException>(
                () => DiscreteActionOutputApplier.Eval(src, dst, m));
        }
    }
}
