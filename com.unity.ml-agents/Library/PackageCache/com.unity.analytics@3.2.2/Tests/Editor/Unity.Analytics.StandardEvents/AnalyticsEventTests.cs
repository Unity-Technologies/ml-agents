#pragma warning disable 0612, 0618

using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace UnityEngine.Analytics.Tests
{
    [TestFixture, Category("Standard Event SDK")]
    public partial class AnalyticsEventTests
    {
        readonly Dictionary<string, object> m_CustomData = new Dictionary<string, object>();
        AnalyticsResult m_Result = AnalyticsResult.Ok;

        [SetUp]
        public void TestCaseSetUp()
        {
            m_Result = AnalyticsResult.Ok;

            m_CustomData.Clear();
            m_CustomData.Add("custom_param", "test");
        }

        [Test]
        public void SdkVersion_FormatTest()
        {
            int major, minor, patch;
            var versions = AnalyticsEvent.sdkVersion.Split('.');

            Assert.AreEqual(3, versions.Length, "Number of integer fields in version format");

            Assert.IsTrue(int.TryParse(versions[0], out major), "Major version is an integer");
            Assert.IsTrue(int.TryParse(versions[1], out minor), "Minor version is an integer");
            Assert.IsTrue(int.TryParse(versions[2], out patch), "Patch version is an integer");

            Assert.LessOrEqual(0, major, "Major version");
            Assert.LessOrEqual(0, minor, "Minor version");
            Assert.LessOrEqual(0, patch, "Patch version");
        }

        [Test]
        public void Custom_EventNameTest(
            [Values("custom_event", "", null)] string eventName
            )
        {
            if (string.IsNullOrEmpty(eventName))
            {
                Assert.Throws<ArgumentException>(() => m_Result = AnalyticsEvent.Custom(eventName));
            }
            else
            {
                Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.Custom(eventName));
                EvaluateAnalyticsResult(m_Result);
            }
        }

        [Test]
        public void Custom_EventDataTest()
        {
            var eventName = "custom_event";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.Custom(eventName, m_CustomData));
            EvaluateCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);
        }

        [Test]
        public void Custom_RegisterUnregisterUnnamedTest()
        {
            Action<IDictionary<string, object>> myAction =
                eventData => eventData.Add("my_key", "my_value");

            AnalyticsEvent.Register(myAction); // Registering for a named AnalyticsEvent

            var eventName = "custom_event";

            Assert.DoesNotThrow(() => m_Result = AnalyticsEvent.Custom(eventName, m_CustomData));

            EvaluateRegisteredCustomData(m_CustomData);
            EvaluateAnalyticsResult(m_Result);

            AnalyticsEvent.Unregister(myAction);
        }

        /// Normal. Unregistered.
        public static void EvaluateCustomData(IDictionary<string, object> customData)
        {
            Assert.AreEqual(1, customData.Count, "Custom param count");
        }

        /// For Registered case.
        public static void EvaluateRegisteredCustomData(IDictionary<string, object> customData)
        {
            Assert.AreEqual(2, customData.Count, "Custom param count");
        }

        public static void EvaluateAnalyticsResult(AnalyticsResult result)
        {
            switch (result)
            {
                case AnalyticsResult.Ok:
                    break;
                case AnalyticsResult.InvalidData:
                    Assert.Fail("Event data is invalid.");
                    break;
                case AnalyticsResult.TooManyItems:
                    Assert.Fail("Event data consists of too many parameters.");
                    break;
                default:
                    Debug.LogFormat("A result of {0} is passable for the purpose of this test.", result);
                    break;
            }
        }
    }
}
