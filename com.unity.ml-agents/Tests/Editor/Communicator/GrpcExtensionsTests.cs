using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Actuators;

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
    }
}
