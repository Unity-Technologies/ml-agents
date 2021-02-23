using System.Collections.Generic;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Analytics;
using Unity.MLAgents.Policies;
using UnityEditor;

namespace Unity.MLAgents.Tests.Analytics
{
    [TestFixture]
    public class TrainingAnalyticsTests
    {
        [TestCase("foo?team=42", ExpectedResult = "foo")]
        [TestCase("foo", ExpectedResult = "foo")]
        [TestCase("foo?bar?team=1337", ExpectedResult = "foo?bar")]
        public string TestParseBehaviorName(string fullyQualifiedBehaviorName)
        {
            return TrainingAnalytics.ParseBehaviorName(fullyQualifiedBehaviorName);
        }

        [Test]
        public void TestRemotePolicy()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }

            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                var actionSpec = ActionSpec.MakeContinuous(3);
                var policy = new RemotePolicy(actionSpec, "TestBehavior?team=42");
                policy.RequestDecision(new AgentInfo(), new List<ISensor>());
            }

            Academy.Instance.Dispose();
        }
    }
}
