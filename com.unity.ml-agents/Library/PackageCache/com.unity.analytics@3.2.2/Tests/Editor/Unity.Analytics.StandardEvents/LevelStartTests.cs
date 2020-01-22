using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void LevelStart_LevelIndexTest(
            [Values(-1, 0, 1)] int levelIndex
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelStart(levelIndex));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void LevelStart_LevelNameTest(
            [Values("test_level", "", null)] string levelName
            )
        {
            if (string.IsNullOrEmpty(levelName))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.LevelStart(levelName));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelStart(levelName));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        // [Test]
        // public void LevelStart_LevelIndex_LevelNameTest (
        //     [Values(0)] int levelIndex,
        //     [Values("test_level", "", null)] string levelName
        // )
        // {
        //     Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelStart(levelIndex, levelName));
        //     EvaluateAnalyticsResult(m_Result);
        // }

        [Test]
        public void LevelStart_CustomDataTest()
        {
            var levelIndex = 0;
            var levelName = "test_level";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelStart(levelName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelStart(levelIndex, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
