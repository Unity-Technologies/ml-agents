using NUnit.Framework;
using UnityEngine;

namespace MLAgents.Tests
{
    public class TimerTests
    {
        [Test]
        public void TestNested()
        {
            TimerStack myTimer = TimerStack.Instance;
            myTimer.Reset();

            using (myTimer.Scoped("foo"))
            {
                for (int i = 0; i < 5; i++)
                {
                    using (myTimer.Scoped("bar"))
                    {
                    }
                }
            }

            var rootChildren = myTimer.RootNode.Children;
            Assert.That(rootChildren, Contains.Key("foo"));
            Assert.AreEqual(rootChildren["foo"].NumCalls, 1);

            var fooChildren = rootChildren["foo"].Children;
            Assert.That(fooChildren, Contains.Key("bar"));
            Assert.AreEqual(fooChildren["bar"].NumCalls, 5);

            myTimer.Reset();
            Assert.AreEqual(myTimer.RootNode.Children, null);
        }
    }
}
