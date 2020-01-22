using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void AdComplete_RewardedTest(
            [Values(true, false)] bool rewarded
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AdComplete(rewarded));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void AdComplete_NetworkStringTest(
            [Values("unityads", "", null)] string network
            )
        {
            var rewarded = true;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AdComplete(rewarded, network));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void AdComplete_NetworkEnumTest(
            [Values(AdvertisingNetwork.UnityAds, AdvertisingNetwork.None)] AdvertisingNetwork network
            )
        {
            var rewarded = true;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AdComplete(rewarded, network));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void AdComplete_PlacementIdTest(
            [Values("rewardedVideo", "", null)] string placementId
            )
        {
            var rewarded = true;
            var network = AdvertisingNetwork.UnityAds;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AdComplete(rewarded, network, placementId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void AdComplete_CustomDataTest()
        {
            var rewarded = true;
            var network = AdvertisingNetwork.UnityAds;
            var placementId = "rewardedVideo";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AdComplete(rewarded, network, placementId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
