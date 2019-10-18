using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using System.IO.Abstractions.TestingHelpers;

namespace MLAgents.Tests
{
    public class DemonstrationTests : MonoBehaviour
    {
        private const string k_DemoDirecory = "Assets/Demonstrations/";
        private const string k_ExtensionType = ".demo";
        private const string k_DemoName = "Test";

        [Test]
        public void TestSanitization()
        {
            const string dirtyString = "abc1234567&!@";
            const string knownCleanString = "abc123";
            var cleanString = DemonstrationRecorder.SanitizeName(dirtyString, 6);
            Assert.AreNotEqual(dirtyString, cleanString);
            Assert.AreEqual(cleanString, knownCleanString);
        }

        [Test]
        public void TestStoreInitalize()
        {
            var fileSystem = new MockFileSystem();
            var demoStore = new DemonstrationStore(fileSystem);

            Assert.IsFalse(fileSystem.Directory.Exists(k_DemoDirecory));

            var brainParameters = new BrainParameters
            {
                vectorObservationSize = 3,
                numStackedVectorObservations = 2,
                vectorActionDescriptions = new[] {"TestActionA", "TestActionB"},
                vectorActionSize = new[] {2, 2},
                vectorActionSpaceType = SpaceType.Discrete
            };

            demoStore.Initialize(k_DemoName, brainParameters, "TestBrain");

            Assert.IsTrue(fileSystem.Directory.Exists(k_DemoDirecory));
            Assert.IsTrue(fileSystem.FileExists(k_DemoDirecory + k_DemoName + k_ExtensionType));

            var agentInfo = new AgentInfo
            {
                reward = 1f,
                actionMasks = new[] {false, true},
                done = true,
                id = 5,
                maxStepReached = true,
                memories = new List<float>(),
                stackedVectorObservation = new List<float>() {1f, 1f, 1f},
                storedTextActions = "TestAction",
                storedVectorActions = new[] {0f, 1f},
                textObservation = "TestAction",
            };

            demoStore.Record(agentInfo);
            demoStore.Close();
        }
    }
}
