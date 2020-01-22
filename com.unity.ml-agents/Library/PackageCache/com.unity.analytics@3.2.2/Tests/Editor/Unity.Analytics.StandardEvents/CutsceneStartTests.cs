using System;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    public partial class AnalyticsEventTests
    {
        [Test]
        public void CutsceneStart_CutsceneNameTest(
            [Values("test_cutscene", "", null)] string name
            )
        {
            if (string.IsNullOrEmpty(name))
            {
                Assert.Throws<ArgumentException>(() => AnalyticsEvent.CutsceneStart(name));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.CutsceneStart(name));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void CutsceneStart_CustomDataTest()
        {
            var name = "test_cutscene";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.CutsceneStart(name, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }
    }
}
