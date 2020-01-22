using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void ChatMessageSent_NoArgsTest()
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ChatMessageSent());
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ChatMessageSent_CustomDataTest()
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ChatMessageSent(m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
