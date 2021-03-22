using NUnit.Framework;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class CompressionSpecTests
    {
        [Test]
        public void TestIsTrivialMapping()
        {
            Assert.IsTrue(CompressionSpec.Default().IsTrivialMapping());

            var spec = new CompressionSpec(SensorCompressionType.PNG, null);
            Assert.AreEqual(spec.IsTrivialMapping(), true);

            spec = new CompressionSpec(SensorCompressionType.PNG, new[] { 0, 0, 0 });
            Assert.AreEqual(spec.IsTrivialMapping(), true);

            spec = new CompressionSpec(SensorCompressionType.PNG, new[] { 0, 1, 2, 3, 4 });
            Assert.AreEqual(spec.IsTrivialMapping(), true);

            spec = new CompressionSpec(SensorCompressionType.PNG, new[] { 1, 2, 3, 4, -1, -1 });
            Assert.AreEqual(spec.IsTrivialMapping(), false);

            spec = new CompressionSpec(SensorCompressionType.PNG, new[] { 0, 0, 0, 1, 1, 1 });
            Assert.AreEqual(spec.IsTrivialMapping(), false);
        }
    }
}
