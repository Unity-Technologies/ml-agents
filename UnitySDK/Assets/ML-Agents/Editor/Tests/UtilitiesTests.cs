using NUnit.Framework;
using UnityEngine;

namespace MLAgents.Tests
{
    public class UtilitiesTests
    {
        [Test]
        public void TestCumSum()
        {
            var output = Utilities.CumSum(new int[]{1, 2, 3, 10});
            CollectionAssert.AreEqual(output, new int[] {0, 1, 3, 6, 16});

            output = Utilities.CumSum(new int[0]);
            CollectionAssert.AreEqual(output, new int[]{0});
            
            output = Utilities.CumSum(new int[]{100});
            CollectionAssert.AreEqual(output, new int[]{0, 100});
            
            output = Utilities.CumSum(new int[]{-1, 10});
            CollectionAssert.AreEqual(output, new int[]{0, -1, 9});
        }
    }
}
