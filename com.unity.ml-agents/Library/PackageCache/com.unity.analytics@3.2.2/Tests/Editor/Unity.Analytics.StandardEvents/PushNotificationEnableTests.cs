using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void PushNotificationEnable_NoArgsTest()
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.PushNotificationEnable());
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void PushNotificationEnable_CustomDataTest()
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.PushNotificationEnable(m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
