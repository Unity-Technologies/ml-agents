using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void LevelSkip_LevelIndexTest(
            [Values(-1, 0, 1)] int levelIndex
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelSkip(levelIndex));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void LevelSkip_LevelNameTest(
            [Values("test_level", "", null)] string levelName
            )
        {
            if (string.IsNullOrEmpty(levelName))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.LevelSkip(levelName));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelSkip(levelName));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        // [Test]
        // public void LevelSkip_LevelIndex_LevelNameTest (
        //     [Values(-1, 0, 1)] int levelIndex,
        //     [Values("test_level", "", null)] string levelName
        // )
        // {
        //     Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelSkip(levelIndex, levelName));
        //     EvaluateAnalyticsResult(m_Result);
        // }

        [Test]
        public void LevelSkip_CustomDataTest()
        {
            var levelIndex = 0;
            var levelName = "test_level";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelSkip(levelName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelSkip(levelIndex, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
