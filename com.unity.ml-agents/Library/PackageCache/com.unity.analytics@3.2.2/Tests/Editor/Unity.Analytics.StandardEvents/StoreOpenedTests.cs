using System.Collections.Generic;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void StoreOpened_StoreTypeTest(
            [Values(StoreType.Premium, StoreType.Soft)] StoreType storeType
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.StoreOpened(storeType));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void StoreOpened_CustomDataTest()
        {
            var storeType = StoreType.Soft;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.StoreOpened(storeType, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
