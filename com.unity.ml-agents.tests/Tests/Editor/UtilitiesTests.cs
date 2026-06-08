using NUnit.Framework;

namespace Unity.MLAgents.Tests
{
    public class UtilitiesTests
    {
        [Test]
        public void TestCumSum()
        {
            var output = Utilities.CumSum(new[] { 1, 2, 3, 10 });
            CollectionAssert.AreEqual(output, new[] { 0, 1, 3, 6, 16 });

            output = Utilities.CumSum(new int[0]);
            CollectionAssert.AreEqual(output, new[] { 0 });

            output = Utilities.CumSum(new[] { 100 });
            CollectionAssert.AreEqual(output, new[] { 0, 100 });

            output = Utilities.CumSum(new[] { -1, 10 });
            CollectionAssert.AreEqual(output, new[] { 0, -1, 9 });
        }
    }
}
