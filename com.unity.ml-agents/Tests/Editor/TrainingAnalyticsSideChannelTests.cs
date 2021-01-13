using System;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Google.Protobuf;
using Unity.MLAgents.Analytics;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.CommunicatorObjects;


namespace Unity.MLAgents.Tests
{
    public class TrainingAnalyticsSideChannelTests
    {
        [Test]
        public void TestTrainingEnvironmentReceived()
        {
            var anyMsg = Google.Protobuf.WellKnownTypes.Any.Pack(new TrainingEnvironmentInitialized());
            var anyMsgBytes = anyMsg.ToByteArray();
            var sideChannel = new TrainingAnalyticsSideChannel();
            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                sideChannel.ProcessMessage(anyMsgBytes);
            }
        }

        [Test]
        public void TestTrainingBehaviorReceived()
        {
            var anyMsg = Google.Protobuf.WellKnownTypes.Any.Pack(new TrainingBehaviorInitialized());
            var anyMsgBytes = anyMsg.ToByteArray();
            var sideChannel = new TrainingAnalyticsSideChannel();
            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                sideChannel.ProcessMessage(anyMsgBytes);
            }
        }

        [Test]
        public void TestInvalidProtobufMessage()
        {
            var badBytes = Encoding.ASCII.GetBytes("Lorem ipsum");
            var sideChannel = new TrainingAnalyticsSideChannel();
            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                sideChannel.ProcessMessage(badBytes);
            }

            var anyMsg = Google.Protobuf.WellKnownTypes.Any.Pack(new TrainingBehaviorInitialized());
            var anyMsgBytes = anyMsg.ToByteArray();
            var truncatedMessage = new ArraySegment<byte>(anyMsgBytes, 0, anyMsgBytes.Length - 1).ToArray();
            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                sideChannel.ProcessMessage(truncatedMessage);
            }
        }
    }
}
