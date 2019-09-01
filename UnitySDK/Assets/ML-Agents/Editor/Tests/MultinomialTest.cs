using NUnit.Framework;
using MLAgents.InferenceBrain.Utils;

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
        public void TestDim3()
        {
            var m = new Multinomial(2018);
            var cdf = new[] {0.1f, 0.2f, 0.7f};

            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(2, m.Sample(cdf));
            Assert.AreEqual(1, m.Sample(cdf));
        }
    }
}
