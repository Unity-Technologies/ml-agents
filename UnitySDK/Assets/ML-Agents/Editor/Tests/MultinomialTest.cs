using System;
using Barracuda;
using NUnit.Framework;
using UnityEngine;
using MLAgents.InferenceBrain;
using MLAgents.InferenceBrain.Utils;

namespace MLAgents.Tests
{
    public class MultinomialTest
    {
        [Test]
        public void TestEvalP()
        {
            Multinomial m = new Multinomial(2018);

            TensorProxy src = new TensorProxy
            {
                Data = new Tensor(1, 3, new[] {0.1f, 0.2f, 0.7f}),
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            TensorProxy dst = new TensorProxy
            {
                Data = new Tensor(1, 3),
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            m.Eval(src, dst);

            float[] reference = {2, 2, 1};
            for (var i = 0; i < dst.Data.length; i++)
            {
                Assert.AreEqual(reference[i], dst.Data[i]);
                ++i;
            }
        }

        [Test]
        public void TestEvalLogits()
        {
            Multinomial m = new Multinomial(2018);

            TensorProxy src = new TensorProxy
            {
                Data = new Tensor(1, 3, new[] {Mathf.Log(0.1f) - 50, Mathf.Log(0.2f) - 50, Mathf.Log(0.7f) - 50}),
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            TensorProxy dst = new TensorProxy
            {
                Data = new Tensor(1, 3),
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            m.Eval(src, dst);

            float[] reference = {2, 2, 2};
            for (var i = 0; i < dst.Data.length; i++)
            {
                Assert.AreEqual(reference[i], dst.Data[i]);
                ++i;
            }
        }

        [Test]
        public void TestEvalBatching()
        {
            Multinomial m = new Multinomial(2018);

            TensorProxy src = new TensorProxy
            {
                Data = new Tensor(2, 3, new []
                {
                    Mathf.Log(0.1f) - 50, Mathf.Log(0.2f) - 50, Mathf.Log(0.7f) - 50,
                    Mathf.Log(0.3f) - 25, Mathf.Log(0.4f) - 25, Mathf.Log(0.3f) - 25
                    
                }),
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            TensorProxy dst = new TensorProxy
            {
                Data = new Tensor(2, 3),
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            m.Eval(src, dst);

            float[] reference = {2, 2, 2, 0, 1, 0};
            for (var i = 0; i < dst.Data.length; i++)
            {
                Assert.AreEqual(reference[i], dst.Data[i]);
                ++i;
            }
        }
        
        [Test]
        public void TestSrcInt()
        {
            Multinomial m = new Multinomial(2018);

            TensorProxy src = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.Integer
            };

            Assert.Throws<NotImplementedException>(() => m.Eval(src, null));
        }
        
        [Test]
        public void TestDstInt()
        {
            Multinomial m = new Multinomial(2018);

            TensorProxy src = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint
            };
            TensorProxy dst = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.Integer
            };

            Assert.Throws<ArgumentException>(() => m.Eval(src, dst));
        }
        
        [Test]
        public void TestSrcDataNull()
        {
            Multinomial m = new Multinomial(2018);
            
            TensorProxy src = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint
            };
            TensorProxy dst = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(() => m.Eval(src, dst));
        }

        [Test]
        public void TestDstDataNull()
        {
            Multinomial m = new Multinomial(2018);
            
            TensorProxy src = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint,
                Data = new Tensor(0,1)
            };
            TensorProxy dst = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(() => m.Eval(src, dst));
        }
        
        [Test]
        public void TestUnequalBatchSize()
        {
            Multinomial m = new Multinomial(2018);
            
            TensorProxy src = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint,
                Data = new Tensor(1, 1)
            };
            TensorProxy dst = new TensorProxy
            {
                ValueType = TensorProxy.TensorType.FloatingPoint,
                Data = new Tensor(2, 1)
            };

            Assert.Throws<ArgumentException>(() => m.Eval(src, dst));
        }
        
        
    }
}
