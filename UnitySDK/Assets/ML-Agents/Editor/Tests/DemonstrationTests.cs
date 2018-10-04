using NUnit.Framework;
using UnityEngine;
using System.IO.Abstractions.TestingHelpers;

namespace MLAgents.Tests
{
    public class DemonstrationTests : MonoBehaviour
    {
        private const string DemoDirecory = "Assets/Demonstrations/";
        private const string ExtensionType = ".demo";
        
        [Test]
        public void TestSanitization()
        {
            var dirtyString = "abc123&!@";
            var cleanString = DemonstrationRecorder.SanitizeName(dirtyString);
            Assert.AreNotEqual(dirtyString, cleanString);
            Assert.AreEqual(cleanString, "abc123");
        }

        [Test]
        public void TestStoreInitalize()
        {
            var fileSystem = new MockFileSystem();
            var demoStore = new DemonstrationStore(fileSystem);

            Assert.IsFalse(fileSystem.Directory.Exists(DemoDirecory));
            
            var brainParameters = new BrainParameters
            {
                vectorObservationSize = 8,
                numStackedVectorObservations = 2,
                cameraResolutions = new [] {new resolution()},
                vectorActionDescriptions = new [] {"TestActionA", "TestActionB"},
                vectorActionSize = new [] {2, 2},
                vectorActionSpaceType = SpaceType.discrete
            };
            
            demoStore.Initialize("Test", brainParameters, "TestBrain");
            
            Assert.IsTrue(fileSystem.Directory.Exists(DemoDirecory));
            Assert.IsTrue(fileSystem.FileExists(DemoDirecory + "Test" + ExtensionType));
        }
    }
}
