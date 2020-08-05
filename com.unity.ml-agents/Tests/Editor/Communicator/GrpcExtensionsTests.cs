using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class GrpcExtensionsTests
    {

        public void TestDefaultBrainParametersToProto()
        {
            // Should be able to convert a default instance to proto.
            var brain = new BrainParameters();
            brain.ToProto("foo", false);
        }


        public void TestDefaultAgentInfoToProto()
        {
            // Should be able to convert a default instance to proto.
            var agentInfo = new AgentInfo();
            agentInfo.ToInfoActionPairProto();
            agentInfo.ToAgentInfoProto();
        }


        public void TestDefaultDemonstrationMetaDataToProto()
        {
            // Should be able to convert a default instance to proto.
            var demoMetaData = new DemonstrationMetaData();
            demoMetaData.ToProto();
        }
    }
}
