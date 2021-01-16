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
    /// <summary>
    /// These tests send messages through the event handling code.
    /// There's no output to test, so just make sure there are no exceptions
    /// (and get the code coverage above the minimum).
    /// </summary>
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
            // Test an invalid (non-protobuf) message. This should silently ignore the data.
            var badBytes = Encoding.ASCII.GetBytes("Lorem ipsum");
            var sideChannel = new TrainingAnalyticsSideChannel();
            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                sideChannel.ProcessMessage(badBytes);
            }

            // Test an almost-valid message. This should silently ignore the data.
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
