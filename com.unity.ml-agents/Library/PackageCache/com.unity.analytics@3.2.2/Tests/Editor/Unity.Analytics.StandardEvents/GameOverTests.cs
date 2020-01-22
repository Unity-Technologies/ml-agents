using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void GameOver_NoArgsTest()
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.GameOver());
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void GameOver_LevelIndexTest(
            [Values(-1, 0, 1)] int levelIndex
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.GameOver(levelIndex));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void GameOver_LevelNameTest(
            [Values("test_level", "", null)] string levelName
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.GameOver(levelName));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void GameOver_LevelIndex_LevelNameTest(
            [Values(0)] int levelIndex,
            [Values("test_level", "", null)] string levelName
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.GameOver(levelIndex, levelName));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void GameOver_CustomDataTest()
        {
            var levelIndex = 0;
            var levelName = "test_level";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.GameOver(levelName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.GameOver(levelIndex, levelName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
