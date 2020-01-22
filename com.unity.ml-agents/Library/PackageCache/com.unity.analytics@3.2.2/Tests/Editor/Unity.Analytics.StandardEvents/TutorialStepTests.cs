using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void TutorialStep_StepIndexTest(
            [Values(-1, 0, 1)] int stepIndex
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.TutorialStep(stepIndex));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void TutorialStep_TutorialIdTest(
            [Values("test_tutorial", "", null)] string tutorialId
            )
        {
            var stepIndex = 0;

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.TutorialStep(stepIndex, tutorialId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void TutorialStep_CustomDataTest()
        {
            var stepIndex = 0;
            var tutorialId = "test_tutorial";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.TutorialStep(stepIndex, tutorialId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
