using System.Collections.Generic;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.Barracuda;
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
                var brainParameters = new BrainParameters();
                brainParameters.VectorObservationSize = 8;
                brainParameters.VectorActionSize = new [] { 2 };
                brainParameters.NumStackedVectorObservations = 1;
                brainParameters.VectorActionSpaceType = SpaceType.Continuous;

                var policy = new RemotePolicy(brainParameters, "TestBehavior?team=42");
                policy.RequestDecision(new AgentInfo(), new List<ISensor>());
            }

            Academy.Instance.Dispose();
        }
    }
}
