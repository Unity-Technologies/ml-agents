using System;
using NUnit.Framework;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents.Tests.Actuators
{
    [TestFixture]
    public class ActionSegmentTests
    {
        [Test]
        public void TestConstruction()
        {
            var floatArray = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f };
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new ActionSegment<float>(floatArray, 100, 1));

            var segment = new ActionSegment<float>(Array.Empty<float>(), 0, 0);
            Assert.AreEqual(segment, ActionSegment<float>.Empty);
        }
        [Test]
        public void TestIndexing()
        {
            var floatArray = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f };
            for (var i = 0; i < floatArray.Length; i++)
            {
                var start = 0 + i;
                var length = floatArray.Length - i;
                var actionSegment = new ActionSegment<float>(floatArray, start, length);
                for (var j = 0; j < actionSegment.Length; j++)
                {
                    Assert.AreEqual(actionSegment[j], floatArray[start + j]);
                }
            }
        }

        [Test]
        public void TestEnumerator()
        {
            var floatArray = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f };
            for (var i = 0; i < floatArray.Length; i++)
            {
                var start = 0 + i;
                var length = floatArray.Length - i;
                var actionSegment = new ActionSegment<float>(floatArray, start, length);
                var j = 0;
                foreach (var item in actionSegment)
                {
                    Assert.AreEqual(item, floatArray[start + j++]);
                }
            }
        }

        [Test]
        public void TestNullConstructor()
        {
            var actionSegment = new ActionSegment<float>(null);
            Assert.IsTrue(actionSegment.Length == 0);
            Assert.IsTrue(actionSegment.Array == Array.Empty<float>());
        }

    }

}
