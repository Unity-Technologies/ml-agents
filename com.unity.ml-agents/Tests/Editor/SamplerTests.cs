using System;
using NUnit.Framework;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.SideChannels;

namespace Unity.MLAgents.Tests
{
    public class SamplerTests
    {
        const int k_Seed = 1337;
        const double k_Epsilon = 0.0001;
        EnvironmentParametersChannel m_Channel = new EnvironmentParametersChannel(); 
    
        public SamplerTests()
        {
            SideChannelsManager.RegisterSideChannel(m_Channel);
        }

        [Test]
        public void UniformSamplerTest()
        {
            float min_value = 1.0f;
            float max_value = 2.0f;
            string parameter = "parameter1";
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 1.0f);
            using (var outgoingMsg = new OutgoingMessage())
            {
                outgoingMsg.WriteString(parameter);
                // 1 indicates this meessage is a Sampler
                outgoingMsg.WriteInt32(1);
                outgoingMsg.WriteInt32(k_Seed);
                outgoingMsg.WriteInt32((int)SamplerType.Uniform);
                outgoingMsg.WriteFloat32(min_value);
                outgoingMsg.WriteFloat32(max_value);
                byte[] message = GetByteMessage(m_Channel, outgoingMsg);
                SideChannelsManager.ProcessSideChannelData(message);
            }
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 1.208888f, k_Epsilon);
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 1.118017f, k_Epsilon);
        }

        [Test]
        public void GaussianSamplerTest()
        {
            float mean = 3.0f;
            float stddev = 0.2f;
            string parameter = "parameter2";
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 1.0f);
            using (var outgoingMsg = new OutgoingMessage())
            {
                outgoingMsg.WriteString(parameter);
                // 1 indicates this meessage is a Sampler
                outgoingMsg.WriteInt32(1);
                outgoingMsg.WriteInt32(k_Seed);
                outgoingMsg.WriteInt32((int)SamplerType.Gaussian);
                outgoingMsg.WriteFloat32(mean);
                outgoingMsg.WriteFloat32(stddev);
                byte[] message = GetByteMessage(m_Channel, outgoingMsg);
                SideChannelsManager.ProcessSideChannelData(message);
            }
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 2.936162f, k_Epsilon);
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 2.951348f, k_Epsilon);
        }

        [Test]
        public void MultiRangeUniformSamplerTest()
        {
            float[] intervals = new float[4];
            intervals[0] = 1.8f;
            intervals[1] = 2f;
            intervals[2] = 3.2f;
            intervals[3] = 4.1f;
            string parameter = "parameter3";
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 1.0f);
            using (var outgoingMsg = new OutgoingMessage())
            {
                outgoingMsg.WriteString(parameter);
                // 1 indicates this meessage is a Sampler
                outgoingMsg.WriteInt32(1);
                outgoingMsg.WriteInt32(k_Seed);
                outgoingMsg.WriteInt32((int)SamplerType.MultiRangeUniform);
                outgoingMsg.WriteFloatList(intervals);
                byte[] message = GetByteMessage(m_Channel, outgoingMsg);
                SideChannelsManager.ProcessSideChannelData(message);
            }
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 3.388, k_Epsilon);
            Assert.AreEqual(m_Channel.GetWithDefault(parameter, 1.0f), 1.823603, k_Epsilon);
        }

        internal static byte[] GetByteMessage(SideChannel sideChannel, OutgoingMessage msg)
        {
            byte[] message = msg.ToByteArray();
            using (var memStream = new MemoryStream())
            {
                using (var binaryWriter = new BinaryWriter(memStream))
                {
                    binaryWriter.Write(sideChannel.ChannelId.ToByteArray());
                    binaryWriter.Write(message.Length);
                    binaryWriter.Write(message);
                }
            return memStream.ToArray();
            }
        }
    }
}
