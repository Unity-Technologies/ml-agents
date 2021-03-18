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

            var spec = CompressionSpec.Compressed(SensorCompressionType.PNG);
            spec.CompressedChannelMapping = null;
            Assert.AreEqual(spec.IsTrivialMapping(), true);
            spec.CompressedChannelMapping = new[] { 0, 0, 0 };
            Assert.AreEqual(spec.IsTrivialMapping(), true);
            spec.CompressedChannelMapping = new[] { 0, 1, 2, 3, 4 };
            Assert.AreEqual(spec.IsTrivialMapping(), true);
            spec.CompressedChannelMapping = new[] { 1, 2, 3, 4, -1, -1 };
            Assert.AreEqual(spec.IsTrivialMapping(), false);
            spec.CompressedChannelMapping = new[] { 0, 0, 0, 1, 1, 1 };
            Assert.AreEqual(spec.IsTrivialMapping(), false);
        }
    }
}
