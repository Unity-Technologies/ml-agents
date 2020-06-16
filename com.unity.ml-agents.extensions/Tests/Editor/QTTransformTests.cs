using UnityEngine;
using NUnit.Framework;

namespace Unity.MLAgents.Extensions.Tests
{

    internal class QTTransformTests
    {
        bool ApproxEquals(Quaternion q1, Quaternion q2, float absTolerance = 1e-6f)
        {
            for (var i = 0; i < 4; i++)
            {
                if (Mathf.Abs(q1[i] - q2[i]) > absTolerance)
                {
                    return false;
                }
            }

            return true;
        }

        [Test]
        public void TestInverse()
        {
            QTTransform t = new QTTransform
            {
                Rotation = Quaternion.AngleAxis(23.0f, new Vector3(1, 1, 1).normalized),
                Translation = new Vector3(-1.0f, 2.0f, 3.0f)
            };

            var inverseT = t.Inverse();
            var product = inverseT * t;
            Assert.IsTrue(product.Translation.Equals(Vector3.zero));
            Assert.IsTrue(ApproxEquals(Quaternion.identity, product.Rotation));

        }

    }

}
