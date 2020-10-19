using System;
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

        class OneTimeThrower
        {
            RecursionChecker m_checker = new RecursionChecker("OneTimeThrower");
            public int NumCalls;

            public void DoStuff()
            {
                // This method throws from inside the checker the first time.
                // Later calls do nothing.
                NumCalls++;
                using (m_checker.Start())
                {
                    if (NumCalls == 1)
                    {
                        throw new ArgumentException("oops");
                    }
                }
            }
        }

        [Test]
        public void TestThrowResetsFlag()
        {
            var ott = new OneTimeThrower();
            Assert.Throws<ArgumentException>(() =>
            {
                ott.DoStuff();
            });

            // Make sure the flag is cleared if we throw in the "using". Should be able to step subsequently.
            ott.DoStuff();
            ott.DoStuff();
            Assert.AreEqual(3, ott.NumCalls);
        }
    }
}
