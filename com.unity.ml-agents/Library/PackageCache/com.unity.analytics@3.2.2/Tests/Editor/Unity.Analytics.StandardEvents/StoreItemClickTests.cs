using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void StoreItemClick_StoreTypeTest(
            [Values(StoreType.Premium, StoreType.Soft)] StoreType storeType
            )
        {
            var itemId = "test_item";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.StoreItemClick(storeType, itemId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void StoreItemClick_ItemIdTest(
            [Values("test_item", "", null)] string itemId
            )
        {
            var storeType = StoreType.Soft;

            if (string.IsNullOrEmpty(itemId))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.StoreItemClick(storeType, itemId));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.StoreItemClick(storeType, itemId));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void StoreItemClick_ItemId_ItemNameTest(
            [Values("test_item_id", "", null)] string itemId,
            [Values("Test Item Name", "", null)] string itemName
            )
        {
            var storeType = StoreType.Soft;

            if (string.IsNullOrEmpty(itemId) && string.IsNullOrEmpty(itemName))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.StoreItemClick(storeType, itemId));
            }
            else
            {
                if (string.IsNullOrEmpty(itemId))
                {
                    Assert.Throws<ArgumentException>(() => AnalyticsEvent.StoreItemClick(storeType, itemId));
                }
                else
                {
                    Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.StoreItemClick(storeType, itemId, itemName));
                    EvaluateAnalyticsResult(m_Result);
                }
            }
        }

        [Test]
        public void StoreItemClick_CustomDataTest()
        {
            var storeType = StoreType.Soft;
            var itemId = "test_item";
            var itemName = "Test Item";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.StoreItemClick(storeType, itemId, itemName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
