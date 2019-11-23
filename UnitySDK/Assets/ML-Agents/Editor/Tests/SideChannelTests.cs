using System;
using NUnit.Framework;
using MLAgents;
using System.Collections.Generic;
using System.Text;

namespace MLAgents.Tests
{
    public class SideChannelTests
    {

        // This test side channel only deals in integers
        public class TestSideChannel : SideChannel
        {

            public List<int> m_MessagesReceived = new List<int>();

            public override int ChannelType() { return -1; }

            public override void OnMessageReceived(byte[] data)
            {
                m_MessagesReceived.Add(BitConverter.ToInt32(data, 0));
            }

            public void SendInt(int data)
            {
                QueueMessageToSend(BitConverter.GetBytes(data));
            }
        }

        [Test]
        public void TestIntegerSideChannel()
        {
            var intSender = new TestSideChannel();
            var intReceiver = new TestSideChannel();
            var dictSender = new Dictionary<int, SideChannel> { { intSender.ChannelType(), intSender } };
            var dictReceiver = new Dictionary<int, SideChannel> { { intReceiver.ChannelType(), intReceiver } };

            intSender.SendInt(4);
            intSender.SendInt(5);
            intSender.SendInt(6);

            byte[] fakeData = RpcCommunicator.GetSideChannelMessage(dictSender);
            RpcCommunicator.SendSideChannelData(dictReceiver, fakeData);

            Assert.AreEqual(intReceiver.m_MessagesReceived[0], 4);
            Assert.AreEqual(intReceiver.m_MessagesReceived[1], 5);
            Assert.AreEqual(intReceiver.m_MessagesReceived[2], 6);
        }

        [Test]
        public void TestRawBytesSideChannel()
        {
            var str1 = "Test string";
            var str2 = "Test string, second";

            var strSender = new RawBytesChannel();
            var strReceiver = new RawBytesChannel();
            var dictSender = new Dictionary<int, SideChannel> { { strSender.ChannelType(), strSender } };
            var dictReceiver = new Dictionary<int, SideChannel> { { strReceiver.ChannelType(), strReceiver } };

            strSender.SendRawBytes(Encoding.ASCII.GetBytes(str1));
            strSender.SendRawBytes(Encoding.ASCII.GetBytes(str2));

            byte[] fakeData = RpcCommunicator.GetSideChannelMessage(dictSender);
            RpcCommunicator.SendSideChannelData(dictReceiver, fakeData);

            var messages = strReceiver.ReceiveRawBytes();

            Assert.AreEqual(messages.Count, 2);
            Assert.AreEqual(Encoding.ASCII.GetString(messages[0]), str1);
            Assert.AreEqual(Encoding.ASCII.GetString(messages[1]), str2);

        }

    }
}
