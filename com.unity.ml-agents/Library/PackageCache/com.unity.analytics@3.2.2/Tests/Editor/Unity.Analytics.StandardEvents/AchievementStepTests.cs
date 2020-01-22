using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void AchievementStep_StepIndexTest(
            [Values(-1, 0, 1)] int stepIndex
            )
        {
            var achievementId = "unit_tester";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AchievementStep(stepIndex, achievementId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void AchievementStep_AchievementIdTest(
            [Values("unit_tester", "", null)] string achievementId
            )
        {
            var stepIndex = 0;

            if (string.IsNullOrEmpty(achievementId))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.AchievementStep(stepIndex, achievementId));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AchievementStep(stepIndex, achievementId));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void AchievementStep_CustomDataTest()
        {
            var stepIndex = 0;
            var achievementId = "unit_tester";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.AchievementStep(stepIndex, achievementId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
