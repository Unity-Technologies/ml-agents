using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void LevelQuit_LevelIndexTest(
            [Values(-1, 0, 1)] int levelIndex
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelQuit(levelIndex));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void LevelQuit_LevelNameTest(
            [Values("test_level", "", null)] string levelName
            )
        {
            if (string.IsNullOrEmpty(levelName))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.LevelQuit(levelName));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelQuit(levelName));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        // [Test]
        // public void LevelQuit_LevelIndex_LevelNameTest (
        //     [Values(-1, 0, 1)] int levelIndex,
        //     [Values("test_level", "", null)] string levelName
        // )
        // {
        //     Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelQuit(levelIndex, levelName));
        //     EvaluateAnalyticsResult(m_Result);
        // }

        [Test]
        public void LevelQuit_CustomDataTest()
        {
            var levelIndex = 0;
            var levelName = "test_level";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelQuit(levelName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.LevelQuit(levelIndex, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
