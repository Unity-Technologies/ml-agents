using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void PushNotificationClick_MessageIdTest(
            [Values("test_message", "", null)] string messageId
            )
        {
            if (string.IsNullOrEmpty(messageId))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.PushNotificationClick(messageId));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.PushNotificationClick(messageId));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void PushNotificationClick_CustomDataTest()
        {
            var messageId = "test_message";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.PushNotificationClick(messageId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
