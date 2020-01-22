using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void ScreenVisit_ScreenNameStringTest(
            [Values("test_screen", "", null)] string screenName
            )
        {
            if (string.IsNullOrEmpty(screenName))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.ScreenVisit(screenName));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ScreenVisit(screenName));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void ScreenVisit_ScreenNameEnumTest(
            [Values(ScreenName.CrossPromo, ScreenName.IAPPromo, ScreenName.None)] ScreenName screenName
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ScreenVisit(screenName));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ScreenVisit_CustomDataTest()
        {
            var screenName = ScreenName.MainMenu;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ScreenVisit(screenName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
