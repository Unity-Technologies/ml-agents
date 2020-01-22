using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void TutorialStart_TutorialIdTest(
            [Values("test_tutorial", "", null)] string tutorialId
            )
        {
            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.TutorialStart(tutorialId));
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void TutorialStart_CustomDataTest()
        {
            var tutorialId = "test_tutorial";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.TutorialStart(tutorialId, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
