using NUnit.Framework;
using UnityEngine;
using System.IO.Abstractions.TestingHelpers;
using System.Reflection;
using MLAgents.CommunicatorObjects;
using MLAgents.Sensors;
using MLAgents.Demonstrations;
using MLAgents.Policies;

namespace MLAgents.Tests
{
    [TestFixture]
    public class DemonstrationTests
    {
        const string k_DemoDirectory = "Assets/Demonstrations/";
        const string k_ExtensionType = ".demo";
        const string k_DemoName = "Test";

        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

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
        public void TestStoreInitialize()
        {
            var fileSystem = new MockFileSystem();

            var gameobj = new GameObject("gameObj");

            var bp = gameobj.AddComponent<BehaviorParameters>();
            bp.brainParameters.vectorObservationSize = 3;
            bp.brainParameters.numStackedVectorObservations = 2;
            bp.brainParameters.vectorActionDescriptions = new[] { "TestActionA", "TestActionB" };
            bp.brainParameters.vectorActionSize = new[] { 2, 2 };
            bp.brainParameters.vectorActionSpaceType = SpaceType.Discrete;

            var agent = gameobj.AddComponent<TestAgent>();

            Assert.IsFalse(fileSystem.Directory.Exists(k_DemoDirectory));

            var demoRec = gameobj.AddComponent<DemonstrationRecorder>();
            demoRec.record = true;
            demoRec.demonstrationName = k_DemoName;
            demoRec.demonstrationDirectory = k_DemoDirectory;
            var demoWriter = demoRec.LazyInitialize(fileSystem);

            Assert.IsTrue(fileSystem.Directory.Exists(k_DemoDirectory));
            Assert.IsTrue(fileSystem.FileExists(k_DemoDirectory + k_DemoName + k_ExtensionType));

            var agentInfo = new AgentInfo
            {
                reward = 1f,
                discreteActionMasks = new[] { false, true },
                done = true,
                episodeId = 5,
                maxStepReached = true,
                storedVectorActions = new[] { 0f, 1f },
            };


            demoWriter.Record(agentInfo, new System.Collections.Generic.List<ISensor>());
            demoRec.Close();

            // Make sure close can be called multiple times
            demoWriter.Close();
            demoRec.Close();

            // Make sure trying to write after closing doesn't raise an error.
            demoWriter.Record(agentInfo, new System.Collections.Generic.List<ISensor>());
        }

        public class ObservationAgent : TestAgent
        {
            public override void CollectObservations(VectorSensor sensor)
            {
                collectObservationsCalls += 1;
                sensor.AddObservation(1f);
                sensor.AddObservation(2f);
                sensor.AddObservation(3f);
            }
        }

        [Test]
        public void TestAgentWrite()
        {
            var agentGo1 = new GameObject("TestAgent");
            var bpA = agentGo1.AddComponent<BehaviorParameters>();
            bpA.brainParameters.vectorObservationSize = 3;
            bpA.brainParameters.numStackedVectorObservations = 1;
            bpA.brainParameters.vectorActionDescriptions = new[] { "TestActionA", "TestActionB" };
            bpA.brainParameters.vectorActionSize = new[] { 2, 2 };
            bpA.brainParameters.vectorActionSpaceType = SpaceType.Discrete;

            agentGo1.AddComponent<ObservationAgent>();
            var agent1 = agentGo1.GetComponent<ObservationAgent>();

            agentGo1.AddComponent<DemonstrationRecorder>();
            var demoRecorder = agentGo1.GetComponent<DemonstrationRecorder>();
            var fileSystem = new MockFileSystem();
            demoRecorder.demonstrationDirectory = k_DemoDirectory;
            demoRecorder.demonstrationName = "TestBrain";
            demoRecorder.record = true;
            demoRecorder.LazyInitialize(fileSystem);

            var agentEnableMethod = typeof(Agent).GetMethod("OnEnable",
                BindingFlags.Instance | BindingFlags.NonPublic);
            var agentSendInfo = typeof(Agent).GetMethod("SendInfo",
                BindingFlags.Instance | BindingFlags.NonPublic);

            agentEnableMethod?.Invoke(agent1, new object[] {});

            // Step the agent
            agent1.RequestDecision();
            agentSendInfo?.Invoke(agent1, new object[] {});

            demoRecorder.Close();

            // Read back the demo file and make sure observations were written
            var reader = fileSystem.File.OpenRead("Assets/Demonstrations/TestBrain.demo");
            reader.Seek(DemonstrationWriter.MetaDataBytes + 1, 0);
            BrainParametersProto.Parser.ParseDelimitedFrom(reader);

            var agentInfoProto = AgentInfoActionPairProto.Parser.ParseDelimitedFrom(reader).AgentInfo;
            var obs = agentInfoProto.Observations[2]; // skip dummy sensors
            {
                var vecObs = obs.FloatData.Data;
                Assert.AreEqual(bpA.brainParameters.vectorObservationSize, vecObs.Count);
                for (var i = 0; i < vecObs.Count; i++)
                {
                    Assert.AreEqual((float)i + 1, vecObs[i]);
                }
            }
        }
    }
}
