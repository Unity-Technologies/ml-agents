using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void IAPTransaction_ContextTest(
            [Values("test", "", null)] string context)
        {
            var price = 1f;
            var itemId = "test_item";

            if (string.IsNullOrEmpty(context))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.IAPTransaction(context, price, itemId));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.IAPTransaction(context, price, itemId));
            }

            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void IAPTransaction_PriceTest(
            [Values(-1f, 0f, 1f)] float price)
        {
            var context = "test";
            var itemId = "test_item";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.IAPTransaction(context, price, itemId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void IAPTransaction_ItemIdTest(
            [Values("test_item", "", null)] string itemId)
        {
            var context = "test";
            var price = 1f;

            if (string.IsNullOrEmpty(itemId))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.IAPTransaction(context, price, itemId));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.IAPTransaction(context, price, itemId));
            }

            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void IAPTransaction_ItemTypeTest(
            [Values("test_type", "", null)] string itemType)
        {
            var context = "test";
            var price = 1f;
            var itemId = "test_item";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.IAPTransaction(context, price, itemId, itemType));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void IAPTransaction_LevelTest(
            [Values("test_level", "", null)] string level)
        {
            var context = "test";
            var price = 1f;
            var itemId = "test_item";
            var itemType = "test_type";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.IAPTransaction(context, price, itemId, itemType, level));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void IAPTransaction_TransactionIdTest(
            [Values("test_id", "", null)] string transactionId)
        {
            var context = "test";
            var price = 1f;
            var itemId = "test_item";
            var itemType = "test_type";
            var level = "test_level";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.IAPTransaction(context, price, itemId, itemType, level, transactionId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void IAPTransaction_CustomDataTest()
        {
            var context = "test";
            var price = 1f;
            var itemId = "test_item";
            var itemType = "test_type";
            var level = "test_level";
            var transactionId = "test_id";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.IAPTransaction(context, price, itemId, itemType, level, transactionId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
