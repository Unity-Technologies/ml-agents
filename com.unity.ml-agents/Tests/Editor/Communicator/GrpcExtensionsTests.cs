using NUnit.Framework;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class GrpcExtensionsTests
    {
        [Test]
        public void TestDefaultBrainParametersToProto()
        {
            // Should be able to convert a default instance to proto.
            var brain = new BrainParameters();
            brain.ToProto("foo", false);
        }

        [Test]
        public void TestDefaultActionSpecToProto()
        {
            // Should be able to convert a default instance to proto.
            var actionSpec = new ActionSpec();
            actionSpec.ToBrainParametersProto("foo", false);

            // Continuous
            actionSpec = ActionSpec.MakeContinuous(3);
            actionSpec.ToBrainParametersProto("foo", false);

            // Discrete
            actionSpec = ActionSpec.MakeDiscrete(1, 2, 3);
            actionSpec.ToBrainParametersProto("foo", false);
        }

        [Test]
        public void TestDefaultAgentInfoToProto()
        {
            // Should be able to convert a default instance to proto.
            var agentInfo = new AgentInfo();
            agentInfo.ToInfoActionPairProto();
            agentInfo.ToAgentInfoProto();
        }

        [Test]
        public void TestDefaultDemonstrationMetaDataToProto()
        {
            // Should be able to convert a default instance to proto.
            var demoMetaData = new DemonstrationMetaData();
            demoMetaData.ToProto();
        }

        class DummySensor : ISensor
        {
            public int[] Shape;
            public SensorCompressionType CompressionType;

            internal DummySensor()
            {
            }

            public int[] GetObservationShape()
            {
                return Shape;
            }

            public int Write(ObservationWriter writer)
            {
                return 0;
            }

            public byte[] GetCompressedObservation()
            {
                return new byte[] { 13, 37 };
            }

            public void Update() { }

            public void Reset() { }

            public SensorCompressionType GetCompressionType()
            {
                return CompressionType;
            }

            public string GetName()
            {
                return "Dummy";
            }
        }

        class DummySparseChannelSensor : DummySensor, ISparseChannelSensor
        {
            public int[] Mapping;
            internal DummySparseChannelSensor()
            {
            }

            public int[] GetCompressedChannelMapping()
            {
                return Mapping;
            }
        }

        [Test]
        public void TestGetObservationProtoCapabilities()
        {
            // Shape, compression type, concatenatedPngObservations, expect throw
            var variants = new[]
            {
                // Vector observations
                (new[] {3}, SensorCompressionType.None, false, false),
                // Uncompressed floats
                (new[] {4, 4, 3}, SensorCompressionType.None, false, false),
                // Compressed floats, 3 channels
                (new[] {4, 4, 3}, SensorCompressionType.PNG, false, true),

                // Compressed floats, >3 channels
                (new[] {4, 4, 4}, SensorCompressionType.PNG, false, false), // Unsupported - results in uncompressed
                (new[] {4, 4, 4}, SensorCompressionType.PNG, true, true), // Supported compressed
            };

            foreach (var (shape, compressionType, supportsMultiPngObs, expectCompressed) in variants)
            {
                var dummySensor = new DummySensor();
                var obsWriter = new ObservationWriter();

                dummySensor.Shape = shape;
                dummySensor.CompressionType = compressionType;
                obsWriter.SetTarget(new float[128], shape, 0);

                var caps = new UnityRLCapabilities
                {
                    ConcatenatedPngObservations = supportsMultiPngObs
                };
                Academy.Instance.TrainerCapabilities = caps;


                var obsProto = dummySensor.GetObservationProto(obsWriter);
                if (expectCompressed)
                {
                    Assert.Greater(obsProto.CompressedData.Length, 0);
                    Assert.AreEqual(obsProto.FloatData, null);
                }
                else
                {
                    Assert.Greater(obsProto.FloatData.Data.Count, 0);
                    Assert.AreEqual(obsProto.CompressedData.Length, 0);
                }
            }


        }

        [Test]
        public void TestIsTrivialMapping()
        {
            Assert.AreEqual(GrpcExtensions.IsTrivialMapping(new DummySensor()), true);

            var sparseChannelSensor = new DummySparseChannelSensor();
            sparseChannelSensor.Mapping = null;
            Assert.AreEqual(GrpcExtensions.IsTrivialMapping(sparseChannelSensor), true);
            sparseChannelSensor.Mapping = new[] { 0, 0, 0 };
            Assert.AreEqual(GrpcExtensions.IsTrivialMapping(sparseChannelSensor), true);
            sparseChannelSensor.Mapping = new[] { 0, 1, 2, 3, 4 };
            Assert.AreEqual(GrpcExtensions.IsTrivialMapping(sparseChannelSensor), true);
            sparseChannelSensor.Mapping = new[] { 1, 2, 3, 4, -1, -1 };
            Assert.AreEqual(GrpcExtensions.IsTrivialMapping(sparseChannelSensor), false);
            sparseChannelSensor.Mapping = new[] { 0, 0, 0, 1, 1, 1 };
            Assert.AreEqual(GrpcExtensions.IsTrivialMapping(sparseChannelSensor), false);
        }
    }
}
