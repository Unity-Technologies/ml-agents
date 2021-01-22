using System;
using NUnit.Framework;
using System.Collections.Generic;
using System.Text;
using Unity.MLAgents.SideChannels;

namespace Unity.MLAgents.Tests
{
    public class SideChannelTests
    {
        // This test side channel only deals in integers
        public class TestSideChannel : SideChannel
        {
            public List<int> messagesReceived = new List<int>();

            public TestSideChannel()
            {
                ChannelId = new Guid("6afa2c06-4f82-11ea-b238-784f4387d1f7");
            }

            protected override void OnMessageReceived(IncomingMessage msg)
            {
                messagesReceived.Add(msg.ReadInt32());
            }

            public void SendInt(int value)
            {
                using (var msg = new OutgoingMessage())
                {
                    msg.WriteInt32(value);
                    QueueMessageToSend(msg);
                }
            }
        }

        [Test]
        public void TestIntegerSideChannel()
        {
            var intSender = new TestSideChannel();
            var intReceiver = new TestSideChannel();
            var dictSender = new Dictionary<Guid, SideChannel> { { intSender.ChannelId, intSender } };
            var dictReceiver = new Dictionary<Guid, SideChannel> { { intReceiver.ChannelId, intReceiver } };

            intSender.SendInt(4);
            intSender.SendInt(5);
            intSender.SendInt(6);

            byte[] fakeData = SideChannelManager.GetSideChannelMessage(dictSender);
            SideChannelManager.ProcessSideChannelData(dictReceiver, fakeData);

            Assert.AreEqual(intReceiver.messagesReceived[0], 4);
            Assert.AreEqual(intReceiver.messagesReceived[1], 5);
            Assert.AreEqual(intReceiver.messagesReceived[2], 6);
        }

        [Test]
        public void TestRawBytesSideChannel()
        {
            var str1 = "Test string";
            var str2 = "Test string, second";

            var strSender = new RawBytesChannel(new Guid("9a5b8954-4f82-11ea-b238-784f4387d1f7"));
            var strReceiver = new RawBytesChannel(new Guid("9a5b8954-4f82-11ea-b238-784f4387d1f7"));
            var dictSender = new Dictionary<Guid, SideChannel> { { strSender.ChannelId, strSender } };
            var dictReceiver = new Dictionary<Guid, SideChannel> { { strReceiver.ChannelId, strReceiver } };

            strSender.SendRawBytes(Encoding.ASCII.GetBytes(str1));
            strSender.SendRawBytes(Encoding.ASCII.GetBytes(str2));

            byte[] fakeData = SideChannelManager.GetSideChannelMessage(dictSender);
            SideChannelManager.ProcessSideChannelData(dictReceiver, fakeData);

            var messages = strReceiver.GetAndClearReceivedMessages();

            Assert.AreEqual(messages.Count, 2);
            Assert.AreEqual(Encoding.ASCII.GetString(messages[0]), str1);
            Assert.AreEqual(Encoding.ASCII.GetString(messages[1]), str2);
        }

        [Test]
        public void TestFloatPropertiesSideChannel()
        {
            var k1 = "gravity";
            var k2 = "length";
            int wasCalled = 0;

            var propA = new FloatPropertiesChannel();
            var propB = new FloatPropertiesChannel();
            var dictReceiver = new Dictionary<Guid, SideChannel> { { propA.ChannelId, propA } };
            var dictSender = new Dictionary<Guid, SideChannel> { { propB.ChannelId, propB } };

            propA.RegisterCallback(k1, f => { wasCalled++; });
            var tmp = propB.GetWithDefault(k2, 3.0f);
            Assert.AreEqual(tmp, 3.0f);
            propB.Set(k2, 1.0f);
            tmp = propB.GetWithDefault(k2, 3.0f);
            Assert.AreEqual(tmp, 1.0f);

            byte[] fakeData = SideChannelManager.GetSideChannelMessage(dictSender);
            SideChannelManager.ProcessSideChannelData(dictReceiver, fakeData);

            tmp = propA.GetWithDefault(k2, 3.0f);
            Assert.AreEqual(tmp, 1.0f);

            Assert.AreEqual(wasCalled, 0);
            propB.Set(k1, 1.0f);
            Assert.AreEqual(wasCalled, 0);
            fakeData = SideChannelManager.GetSideChannelMessage(dictSender);
            SideChannelManager.ProcessSideChannelData(dictReceiver, fakeData);
            Assert.AreEqual(wasCalled, 1);

            var keysA = propA.Keys();
            Assert.AreEqual(2, keysA.Count);
            Assert.IsTrue(keysA.Contains(k1));
            Assert.IsTrue(keysA.Contains(k2));

            var keysB = propA.Keys();
            Assert.AreEqual(2, keysB.Count);
            Assert.IsTrue(keysB.Contains(k1));
            Assert.IsTrue(keysB.Contains(k2));
        }

        [Test]
        public void TestOutgoingMessageRawBytes()
        {
            // Make sure that SetRawBytes resets the buffer correctly.
            // Write 8 bytes (an int and float) then call SetRawBytes with 4 bytes
            var msg = new OutgoingMessage();
            msg.WriteInt32(42);
            msg.WriteFloat32(1.0f);

            var data = new byte[] { 1, 2, 3, 4 };
            msg.SetRawBytes(data);

            var result = msg.ToByteArray();
            Assert.AreEqual(data, result);
        }

        [Test]
        public void TestMessageReadWrites()
        {
            var boolVal = true;
            var intVal = 1337;
            var floatVal = 4.2f;
            var floatListVal = new float[] { 1001, 1002 };
            var stringVal = "mlagents!";

            IncomingMessage incomingMsg;
            using (var outgoingMsg = new OutgoingMessage())
            {
                outgoingMsg.WriteBoolean(boolVal);
                outgoingMsg.WriteInt32(intVal);
                outgoingMsg.WriteFloat32(floatVal);
                outgoingMsg.WriteString(stringVal);
                outgoingMsg.WriteFloatList(floatListVal);

                incomingMsg = new IncomingMessage(outgoingMsg.ToByteArray());
            }

            Assert.AreEqual(boolVal, incomingMsg.ReadBoolean());
            Assert.AreEqual(intVal, incomingMsg.ReadInt32());
            Assert.AreEqual(floatVal, incomingMsg.ReadFloat32());
            Assert.AreEqual(stringVal, incomingMsg.ReadString());
            Assert.AreEqual(floatListVal, incomingMsg.ReadFloatList());
        }

        [Test]
        public void TestMessageReadDefaults()
        {
            // Make sure reading past the end of a message will apply defaults.
            IncomingMessage incomingMsg;
            using (var outgoingMsg = new OutgoingMessage())
            {
                incomingMsg = new IncomingMessage(outgoingMsg.ToByteArray());
            }

            Assert.AreEqual(false, incomingMsg.ReadBoolean());
            Assert.AreEqual(true, incomingMsg.ReadBoolean(defaultValue: true));

            Assert.AreEqual(0, incomingMsg.ReadInt32());
            Assert.AreEqual(42, incomingMsg.ReadInt32(defaultValue: 42));

            Assert.AreEqual(0.0f, incomingMsg.ReadFloat32());
            Assert.AreEqual(1337.0f, incomingMsg.ReadFloat32(defaultValue: 1337.0f));

            Assert.AreEqual(default(string), incomingMsg.ReadString());
            Assert.AreEqual("foo", incomingMsg.ReadString(defaultValue: "foo"));

            Assert.AreEqual(default(float[]), incomingMsg.ReadFloatList());
            Assert.AreEqual(new float[] { 1001, 1002 }, incomingMsg.ReadFloatList(new float[] { 1001, 1002 }));
        }
    }
}
