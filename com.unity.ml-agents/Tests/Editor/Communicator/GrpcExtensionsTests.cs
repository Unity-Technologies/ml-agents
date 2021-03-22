using System;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

using Unity.MLAgents.Analytics;
using Unity.MLAgents.CommunicatorObjects;

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
            public ObservationSpec ObservationSpec;
            public SensorCompressionType CompressionType;

            internal DummySensor()
            {
            }

            public ObservationSpec GetObservationSpec()
            {
                return ObservationSpec;
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

            public CompressionSpec GetCompressionSpec()
            {
                return new CompressionSpec(CompressionType);
            }

            public string GetName()
            {
                return "Dummy";
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
                var inplaceShape = InplaceArray<int>.FromList(shape);
                var dummySensor = new DummySensor();
                var obsWriter = new ObservationWriter();

                if (shape.Length == 1)
                {
                    dummySensor.ObservationSpec = ObservationSpec.Vector(shape[0]);
                }
                else if (shape.Length == 3)
                {
                    dummySensor.ObservationSpec = ObservationSpec.Visual(shape[0], shape[1], shape[2]);
                }
                else
                {
                    throw new ArgumentOutOfRangeException();
                }
                dummySensor.CompressionType = compressionType;
                obsWriter.SetTarget(new float[128], inplaceShape, 0);

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
        public void TestDefaultTrainingEvents()
        {
            var trainingEnvInit = new TrainingEnvironmentInitialized
            {
                PythonVersion = "test",
            };
            var trainingEnvInitEvent = trainingEnvInit.ToTrainingEnvironmentInitializedEvent();
            Assert.AreEqual(trainingEnvInit.PythonVersion, trainingEnvInitEvent.TrainerPythonVersion);

            var trainingBehavInit = new TrainingBehaviorInitialized
            {
                BehaviorName = "testBehavior",
                ExtrinsicRewardEnabled = true,
                CuriosityRewardEnabled = true,

                RecurrentEnabled = true,
                SelfPlayEnabled = true,
            };
            var trainingBehavInitEvent = trainingBehavInit.ToTrainingBehaviorInitializedEvent();
            Assert.AreEqual(trainingBehavInit.BehaviorName, trainingBehavInitEvent.BehaviorName);

            Assert.AreEqual(RewardSignals.Extrinsic | RewardSignals.Curiosity, trainingBehavInitEvent.RewardSignalFlags);
            Assert.AreEqual(TrainingFeatures.Recurrent | TrainingFeatures.SelfPlay, trainingBehavInitEvent.TrainingFeatureFlags);
        }
    }
}
