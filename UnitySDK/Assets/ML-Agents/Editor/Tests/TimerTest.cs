using NUnit.Framework;
using UnityEditor.Graphs;
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
                        myTimer.SetGauge("my_gauge", (float)i);
                    }
                }
            }

            var rootChildren = myTimer.RootNode.Children;
            Assert.That(rootChildren, Contains.Key("foo"));
            Assert.AreEqual(rootChildren["foo"].NumCalls, 1);
            var gauge = myTimer.RootNode.Gauges["my_gauge"];
            Assert.NotNull(gauge);
            Assert.AreEqual(5, gauge.count);
            Assert.AreEqual(0, gauge.minValue);
            Assert.AreEqual(4, gauge.maxValue);
            Assert.AreEqual(4, gauge.value);

            var fooChildren = rootChildren["foo"].Children;
            Assert.That(fooChildren, Contains.Key("bar"));
            Assert.AreEqual(fooChildren["bar"].NumCalls, 5);

            myTimer.Reset();
            Assert.AreEqual(myTimer.RootNode.Children, null);
        }
    }
}
