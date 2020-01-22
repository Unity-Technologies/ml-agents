using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void TutorialComplete_TutorialIdTest(
            [Values("test_tutorial", "", null)] string tutorialId
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.TutorialComplete(tutorialId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void TutorialComplete_CustomDataTest()
        {
            var tutorialId = "test_tutorial";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.TutorialComplete(tutorialId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
