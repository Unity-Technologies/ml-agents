using NUnit.Framework;
using MLAgents.Inference.Utils;

namespace MLAgents.Tests
{
    public class MultinomialTest
    {
        [Test]
        public void TestDim1()
        {
            var m = new Multinomial(2018);
            var cdf = new[] {1f};

            Assert.AreEqual(0, m.Sample(cdf));
            Assert.AreEqual(0, m.Sample(cdf));
            Assert.AreEqual(0, m.Sample(cdf));
        }

        [Test]
        public void TestDim1Unscaled()
        {
            var m = new Multinomial(2018);
            var cdf = new[] {0.1f};

            Assert.AreEqual(0, m.Sample(cdf));
            Assert.AreEqual(0, m.Sample(cdf));
            Assert.AreEqual(0, m.Sample(cdf));
        }

        [Test]
        public void TestDim3()
        {
            var m = new Multinomial(2018);
            var cdf = new[] {0.1f, 0.3f, 1.0f};

            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(1, m.Sample(cdf));
        }

        [Test]
        public void TestDim3Unscaled()
        {
            var m = new Multinomial(2018);
            var cdf = new[] {0.05f, 0.15f, 0.5f};

            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(1, m.Sample(cdf));
        }
    }
}
