using NUnit.Framework;

namespace Unity.MLAgents.Tests
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
                        myTimer.SetGauge("my_gauge", i);
                        myTimer.AddMetadata("i", $"{i}");
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
            Assert.AreEqual("4", myTimer.RootNode.Metadata["i"]);

            var fooChildren = rootChildren["foo"].Children;
            Assert.That(fooChildren, Contains.Key("bar"));
            Assert.AreEqual(fooChildren["bar"].NumCalls, 5);

            myTimer.Reset();
            Assert.AreEqual(myTimer.RootNode.Children, null);
        }

        [Test]
        public void TestGauges()
        {
            TimerStack myTimer = TimerStack.Instance;
            myTimer.Reset();

            // Simple test - adding 1's should keep that for the weighted and running averages.
            myTimer.SetGauge("one", 1.0f);
            var oneNode = myTimer.RootNode.Gauges["one"];
            Assert.AreEqual(oneNode.weightedAverage, 1.0f);
            Assert.AreEqual(oneNode.runningAverage, 1.0f);

            for (int i = 0; i < 10; i++)
            {
                myTimer.SetGauge("one", 1.0f);
            }

            Assert.AreEqual(oneNode.weightedAverage, 1.0f);
            Assert.AreEqual(oneNode.runningAverage, 1.0f);

            // Try some more interesting values
            myTimer.SetGauge("increasing", 1.0f);
            myTimer.SetGauge("increasing", 2.0f);
            myTimer.SetGauge("increasing", 3.0f);

            myTimer.SetGauge("decreasing", 3.0f);
            myTimer.SetGauge("decreasing", 2.0f);
            myTimer.SetGauge("decreasing", 1.0f);
            var increasingNode = myTimer.RootNode.Gauges["increasing"];
            var decreasingNode = myTimer.RootNode.Gauges["decreasing"];

            // Expect the running average to be (roughly) the same,
            // but weighted averages will be biased differently.
            Assert.AreEqual(increasingNode.runningAverage, 2.0f);
            Assert.AreEqual(decreasingNode.runningAverage, 2.0f);

            // The older values are actually weighted more heavily, so we expect the
            // increasing series to have a lower moving average.
            Assert.Less(increasingNode.weightedAverage, decreasingNode.weightedAverage);
        }
    }
}
