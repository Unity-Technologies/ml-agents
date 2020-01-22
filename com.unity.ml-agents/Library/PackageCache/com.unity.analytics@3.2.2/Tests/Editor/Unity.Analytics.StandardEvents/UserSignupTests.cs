using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void UserSignup_AuthorizationNetworkStringTest(
            [Values("test_network", "", null)] string network
            )
        {
            if (string.IsNullOrEmpty(network))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.UserSignup(network));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.UserSignup(network));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void UserSignup_AuthorizationNetworkEnumTest(
            [Values(AuthorizationNetwork.Facebook, AuthorizationNetwork.GameCenter, AuthorizationNetwork.None)] AuthorizationNetwork network
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.UserSignup(network));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void UserSignup_CustomDataTest()
        {
            var network = AuthorizationNetwork.Internal;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.UserSignup(network, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
