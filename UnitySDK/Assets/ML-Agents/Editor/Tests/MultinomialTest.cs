using System;
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

            Tensor src = new Tensor
            {
                Data = new Barracuda.Tensor(1, 3, new[] {0.1f, 0.2f, 0.7f}),
                ValueType = Tensor.TensorType.FloatingPoint
            };

            Tensor dst = new Tensor
            {
                Data = new Barracuda.Tensor(1, 3),
                ValueType = Tensor.TensorType.FloatingPoint
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

            Tensor src = new Tensor
            {
                Data = new Barracuda.Tensor(1, 3, new[] {Mathf.Log(0.1f) - 50, Mathf.Log(0.2f) - 50, Mathf.Log(0.7f) - 50}),
                ValueType = Tensor.TensorType.FloatingPoint
            };

            Tensor dst = new Tensor
            {
                Data = new Barracuda.Tensor(1, 3),
                ValueType = Tensor.TensorType.FloatingPoint
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

            Tensor src = new Tensor
            {
                Data = new Barracuda.Tensor(2, 3, new []
                {
                    Mathf.Log(0.1f) - 50, Mathf.Log(0.2f) - 50, Mathf.Log(0.7f) - 50,
                    Mathf.Log(0.3f) - 25, Mathf.Log(0.4f) - 25, Mathf.Log(0.3f) - 25
                    
                }),
                ValueType = Tensor.TensorType.FloatingPoint
            };

            Tensor dst = new Tensor
            {
                Data = new Barracuda.Tensor(2, 3),
                ValueType = Tensor.TensorType.FloatingPoint
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

            Tensor src = new Tensor
            {
                ValueType = Tensor.TensorType.Integer
            };

            Assert.Throws<NotImplementedException>(() => m.Eval(src, null));
        }
        
        [Test]
        public void TestDstInt()
        {
            Multinomial m = new Multinomial(2018);

            Tensor src = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint
            };
            Tensor dst = new Tensor
            {
                ValueType = Tensor.TensorType.Integer
            };

            Assert.Throws<ArgumentException>(() => m.Eval(src, dst));
        }
        
        [Test]
        public void TestSrcDataNull()
        {
            Multinomial m = new Multinomial(2018);
            
            Tensor src = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint
            };
            Tensor dst = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(() => m.Eval(src, dst));
        }

        [Test]
        public void TestDstDataNull()
        {
            Multinomial m = new Multinomial(2018);
            
            Tensor src = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint,
                Data = new Barracuda.Tensor(0,1)
            };
            Tensor dst = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint
            };

            Assert.Throws<ArgumentNullException>(() => m.Eval(src, dst));
        }
        
        [Test]
        public void TestDstWrongShape()
        {
            Multinomial m = new Multinomial(2018);
            
            Tensor src = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint,
                Data = new Barracuda.Tensor(0,1)
            };
            Tensor dst = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint,
                Data = new Barracuda.Tensor(0,2)
            };

            Assert.Throws<ArgumentException>(() => m.Eval(src, dst));
        }

        [Test]
        public void TestUnequalBatchSize()
        {
            Multinomial m = new Multinomial(2018);
            
            Tensor src = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint,
                Data = new Barracuda.Tensor(1, 1)
            };
            Tensor dst = new Tensor
            {
                ValueType = Tensor.TensorType.FloatingPoint,
                Data = new Barracuda.Tensor(2, 1)
            };

            Assert.Throws<ArgumentException>(() => m.Eval(src, dst));
        }
        
        
    }
}
