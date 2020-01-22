using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void ItemAcquired_CurrencyTypeTest(
            [Values(AcquisitionType.Premium, AcquisitionType.Soft)] AcquisitionType currencyType)
        {
            var context = "test";
            var amount = 1f;
            var itemId = "test_item";
            var balance = 1f;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId));
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ItemAcquired_ContextTest(
            [Values("test", "", null)] string context)
        {
            var currencyType = AcquisitionType.Soft;
            var amount = 1f;
            var itemId = "test_item";
            var balance = 1f;

            if (string.IsNullOrEmpty(context))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId));
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
                EvaluateAnalyticsResult(m_Result);

                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void ItemAcquired_AmountTest(
            [Values(-1f, 0f, 1f)] float amount)
        {
            var currencyType = AcquisitionType.Soft;
            var context = "test";
            var itemId = "test_item";
            var balance = 1f;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId));
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ItemAcquired_ItemIdTest(
            [Values("test_item", "", null)] string itemId)
        {
            var currencyType = AcquisitionType.Soft;
            var context = "test";
            var amount = 1f;
            var balance = 1f;

            if (string.IsNullOrEmpty(itemId))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId));
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId));
                EvaluateAnalyticsResult(m_Result);

                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void ItemAcquired_BalanceTest(
            [Values(-1f, 0, 1f)] float balance)
        {
            var currencyType = AcquisitionType.Soft;
            var context = "test";
            var amount = 1f;
            var itemId = "test_item";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ItemAcquired_ItemTypeTest(
            [Values("test_type", "", null)] string itemType)
        {
            var currencyType = AcquisitionType.Soft;
            var context = "test";
            var amount = 1f;
            var itemId = "test_item";
            var balance = 1f;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, itemType));
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance, itemType));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ItemAcquired_LevelTest(
            [Values("test_level", "", null)] string level)
        {
            var currencyType = AcquisitionType.Soft;
            var context = "test";
            var amount = 1f;
            var itemId = "test_item";
            var balance = 1f;
            var itemType = "test_type";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, itemType, level));
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance, itemType, level));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ItemAcquired_TransactionIdTest(
            [Values("test_id", "", null)] string transactionId)
        {
            var currencyType = AcquisitionType.Soft;
            var context = "test";
            var amount = 1f;
            var itemId = "test_item";
            var balance = 1f;
            var itemType = "test_type";
            var level = "test_level";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, itemType, level, transactionId));
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance, itemType, level, transactionId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void ItemAcquired_CustomDataTest()
        {
            var currencyType = AcquisitionType.Soft;
            var context = "test";
            var amount = 1f;
            var itemId = "test_item";
            var balance = 1f;
            var itemType = "test_type";
            var level = "test_level";
            var transactionId = "test_id";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, itemType, level, transactionId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.ItemAcquired(currencyType, context, amount, itemId, balance, itemType, level, transactionId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
