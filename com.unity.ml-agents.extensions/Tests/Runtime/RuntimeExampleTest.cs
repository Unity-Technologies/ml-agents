using NUnit.Framework;

namespace Unity.MLAgents.Extensions.Tests
{
    internal class RuntimeExampleTest
    {
        [Test]
        public void RuntimeTestMath()
        {
            Assert.AreEqual(2, 1 + 1);
        }
    }
}
