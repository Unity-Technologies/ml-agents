using NUnit.Framework;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace Unity.MLAgents.Tests
{
    public class EngineConfigurationChannelTests
    {
        float m_OldTimeScale = 1.0f;

        [SetUp]
        public void Setup()
        {
            m_OldTimeScale = Time.timeScale;
        }

        [TearDown]
        public void TearDown()
        {
            Time.timeScale = m_OldTimeScale;
        }

        [Test]
        public void TestTimeScaleClamping()
        {
            OutgoingMessage pythonMsg = new OutgoingMessage();
            pythonMsg.WriteInt32((int)EngineConfigurationChannel.ConfigurationType.TimeScale);
            pythonMsg.WriteFloat32(1000f);

            var sideChannel = new EngineConfigurationChannel();
            sideChannel.ProcessMessage(pythonMsg.ToByteArray());

#if UNITY_EDITOR
            // Should be clamped
            Assert.AreEqual(100.0f, Time.timeScale);
#else
            // Not sure we can run this test from a player, but just in case, shouldn't clamp.
            Assert.AreEqual(1000.0f, Time.timeScale);
#endif
        }
    }
}
