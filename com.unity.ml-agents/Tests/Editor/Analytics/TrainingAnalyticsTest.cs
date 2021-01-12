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
    }
}
