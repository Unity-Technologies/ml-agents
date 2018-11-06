using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using System.IO.Abstractions.TestingHelpers;

namespace MLAgents.Tests
{
    public class DemonstrationTests : MonoBehaviour
    {
        private const string DemoDirecory = "Assets/Demonstrations/";
        private const string ExtensionType = ".demo";
        private const string DemoName = "Test";
        
        [Test]
        public void TestSanitization()
        {
            const string dirtyString = "abc123&!@";
            const string knownCleanString = "abc123";
            var cleanString = DemonstrationRecorder.SanitizeName(dirtyString);
            Assert.AreNotEqual(dirtyString, cleanString);
            Assert.AreEqual(cleanString, knownCleanString);
        }

        [Test]
        public void TestStoreInitalize()
        {
            var fileSystem = new MockFileSystem();
            var demoStore = new DemonstrationStore(fileSystem);

            Assert.IsFalse(fileSystem.Directory.Exists(DemoDirecory));
            
            var brainParameters = new BrainParameters
            {
                vectorObservationSize = 3,
                numStackedVectorObservations = 2,
                cameraResolutions = new [] {new Resolution()},
                vectorActionDescriptions = new [] {"TestActionA", "TestActionB"},
                vectorActionSize = new [] {2, 2},
                vectorActionSpaceType = SpaceType.discrete
            };
            
            demoStore.Initialize(DemoName, brainParameters, "TestBrain");
            
            Assert.IsTrue(fileSystem.Directory.Exists(DemoDirecory));
            Assert.IsTrue(fileSystem.FileExists(DemoDirecory + DemoName + ExtensionType));

            var agentInfo = new AgentInfo
            {
                reward = 1f,
                visualObservations = new List<Texture2D>(),
                actionMasks = new []{false, true},
                done = true,
                id = 5,
                maxStepReached = true,
                memories = new List<float>(),
                stackedVectorObservation = new List<float>() {1f, 1f, 1f},
                storedTextActions = "TestAction",
                storedVectorActions = new [] {0f, 1f},
                textObservation = "TestAction",
            };
            
            demoStore.Record(agentInfo);
            demoStore.Close();
        }
    }
}
