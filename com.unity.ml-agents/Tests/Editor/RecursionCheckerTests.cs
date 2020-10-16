using NUnit.Framework;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class RecursionCheckerTests
    {
        class InfiniteRecurser
        {
            RecursionChecker m_checker = new RecursionChecker("InfiniteRecurser");
            public int NumCalls = 0;

            public void Implode()
            {
                NumCalls++;
                using (m_checker.Start())
                {
                    Implode();
                }
            }
        }

        [Test]
        public void TestRecursionCheck()
        {
            var rc = new InfiniteRecurser();
            Assert.Throws<UnityAgentsException>(() =>
            {
                rc.Implode();
            });

            // Should increment twice before bailing out.
            Assert.AreEqual(2, rc.NumCalls);
        }
    }
}
