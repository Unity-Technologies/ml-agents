using UnityEngine;
using NUnit.Framework;
using System.Reflection;
using System.Collections.Generic;

namespace MLAgents.Tests
{
    internal class TestPolicy : IPolicy
    {
        public void RequestDecision(AgentInfo info, List<ISensor> sensors) {}

        public float[] DecideAction() { return new float[0]; }

        public void Dispose() {}
    }

    public class TestAgent : Agent
    {
        internal AgentInfo _Info
        {
            get
            {
                return (AgentInfo)typeof(Agent).GetField("m_Info", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
            }
            set
            {
                typeof(Agent).GetField("m_Info", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(this, value);
            }
        }

        internal void SetPolicy(IPolicy policy)
        {
            typeof(Agent).GetField("m_Brain", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(this, policy);
        }

        public int initializeAgentCalls;
        public int collectObservationsCalls;
        public int collectObservationsCallsSinceLastReset;
        public int agentActionCalls;
        public int agentActionCallsSinceLastReset;
        public int agentResetCalls;
        public override void InitializeAgent()
        {
            initializeAgentCalls += 1;

            // Add in some custom Sensors so we can confirm they get sorted as expected.
            var sensor1 = new TestSensor("testsensor1");
            var sensor2 = new TestSensor("testsensor2");

            sensors.Add(sensor2);
            sensors.Add(sensor1);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            collectObservationsCalls += 1;
            collectObservationsCallsSinceLastReset += 1;
            sensor.AddObservation(0f);
        }

        public override void AgentAction(float[] vectorAction)
        {
            agentActionCalls += 1;
            agentActionCallsSinceLastReset += 1;
            AddReward(0.1f);
        }

        public override void AgentReset()
        {
            agentResetCalls += 1;
            collectObservationsCallsSinceLastReset = 0;
            agentActionCallsSinceLastReset = 0;
        }

        public override float[] Heuristic()
        {
            return new float[0];
        }
    }

    public class TestSensor : ISensor
    {
        public string sensorName;

        public TestSensor(string n)
        {
            sensorName = n;
        }

        public int[] GetObservationShape()
        {
            return new[] { 0 };
        }

        public int Write(WriteAdapter adapter)
        {
            // No-op
            return 0;
        }

        public byte[] GetCompressedObservation()
        {
            return null;
        }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        public string GetName()
        {
            return sensorName;
        }

        public void Update() {}
    }

    [TestFixture]
    public class EditModeTestGeneration
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            var aca = Academy.Instance;
            Assert.AreNotEqual(null, aca);
            Assert.AreEqual(0, aca.EpisodeCount);
            Assert.AreEqual(0, aca.StepCount);
            Assert.AreEqual(0, aca.TotalStepCount);
        }

        [Test]
        public void TestAgent()
        {
            var agentGo = new GameObject("TestAgent");
            agentGo.AddComponent<TestAgent>();
            var agent = agentGo.GetComponent<TestAgent>();
            Assert.AreNotEqual(null, agent);
            Assert.AreEqual(0, agent.initializeAgentCalls);
        }
    }

    [TestFixture]
    public class EditModeTestInitialization
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            Assert.AreEqual(false, Academy.IsInitialized);
            var aca = Academy.Instance;
            Assert.AreEqual(true, Academy.IsInitialized);

            // Check that init is idempotent
            aca.LazyInitialize();
            aca.LazyInitialize();

            Assert.AreEqual(0, aca.EpisodeCount);
            Assert.AreEqual(0, aca.StepCount);
            Assert.AreEqual(0, aca.TotalStepCount);
            Assert.AreNotEqual(null, aca.FloatProperties);

            // Check that Dispose is idempotent
            aca.Dispose();
            Assert.AreEqual(false, Academy.IsInitialized);
            aca.Dispose();
        }

        [Test]
        public void TestAcademyDispose()
        {
            var floatProperties1 = Academy.Instance.FloatProperties;
            Academy.Instance.Dispose();

            var floatProperties2 = Academy.Instance.FloatProperties;
            Academy.Instance.Dispose();

            Assert.AreNotEqual(floatProperties1, floatProperties2);
        }

        [Test]
        public void TestAgent()
        {
            var agentGo1 = new GameObject("TestAgent");
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();

            Assert.AreEqual(0, agent1.agentResetCalls);
            Assert.AreEqual(0, agent2.agentResetCalls);
            Assert.AreEqual(0, agent1.initializeAgentCalls);
            Assert.AreEqual(0, agent2.initializeAgentCalls);
            Assert.AreEqual(0, agent1.agentActionCalls);
            Assert.AreEqual(0, agent2.agentActionCalls);


            agent2.LazyInitialize();
            agent1.LazyInitialize();

            // agent1 was not enabled when the academy started
            // The agents have been initialized
            Assert.AreEqual(0, agent1.agentResetCalls);
            Assert.AreEqual(0, agent2.agentResetCalls);
            Assert.AreEqual(1, agent1.initializeAgentCalls);
            Assert.AreEqual(1, agent2.initializeAgentCalls);
            Assert.AreEqual(0, agent1.agentActionCalls);
            Assert.AreEqual(0, agent2.agentActionCalls);

            // Make sure the Sensors were sorted
            Assert.AreEqual(agent1.sensors[0].GetName(), "testsensor1");
            Assert.AreEqual(agent1.sensors[1].GetName(), "testsensor2");
        }
    }

    [TestFixture]
    public class EditModeTestStep
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            var aca = Academy.Instance;

            var numberReset = 0;
            for (var i = 0; i < 10; i++)
            {
                Assert.AreEqual(numberReset, aca.EpisodeCount);
                Assert.AreEqual(i, aca.StepCount);

                // The reset happens at the beginning of the first step
                if (i == 0)
                {
                    numberReset += 1;
                }
                Academy.Instance.EnvironmentStep();
            }
        }

        [Test]
        public void TestAcademyAutostep()
        {
            var aca = Academy.Instance;
            Assert.IsTrue(aca.AutomaticSteppingEnabled);
            aca.AutomaticSteppingEnabled = false;
            Assert.IsFalse(aca.AutomaticSteppingEnabled);
            aca.AutomaticSteppingEnabled = true;
            Assert.IsTrue(aca.AutomaticSteppingEnabled);
        }

        [Test]
        public void TestAgent()
        {
            var agentGo1 = new GameObject("TestAgent");
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();

            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.Awake();
            // agent1 will take an action at every step and request a decision every 2 steps
            // agent2 will request decisions only when RequestDecision is called

            agent1.LazyInitialize();

            var numberAgent1Reset = 0;
            var numberAgent2Initialization = 0;
            var requestDecision = 0;
            var requestAction = 0;
            for (var i = 0; i < 50; i++)
            {
                Assert.AreEqual(numberAgent1Reset, agent1.agentResetCalls);
                // Agent2 is never reset since initialized after academy
                Assert.AreEqual(0, agent2.agentResetCalls);
                Assert.AreEqual(1, agent1.initializeAgentCalls);
                Assert.AreEqual(numberAgent2Initialization, agent2.initializeAgentCalls);
                Assert.AreEqual(i, agent1.agentActionCalls);
                Assert.AreEqual(requestAction, agent2.agentActionCalls);
                Assert.AreEqual((i + 1) / 2, agent1.collectObservationsCalls);
                Assert.AreEqual(requestDecision, agent2.collectObservationsCalls);
                // Agent 1 resets at the first step
                if (i == 0)
                {
                    numberAgent1Reset += 1;
                }
                //Agent 2 is only initialized at step 2
                if (i == 2)
                {
                    agent2.LazyInitialize();
                    numberAgent2Initialization += 1;
                }

                // We are testing request decision and request actions when called
                // at different intervals
                if ((i % 3 == 0) && (i > 2))
                {
                    //Every 3 steps after agent 2 is initialized, request decision
                    requestDecision += 1;
                    requestAction += 1;
                    agent2.RequestDecision();
                }
                else if ((i % 5 == 0) && (i > 2))
                {
                    // Every 5 steps after agent 2 is initialized, request action
                    requestAction += 1;
                    agent2.RequestAction();
                }
                aca.EnvironmentStep();
            }
        }
    }

    [TestFixture]
    public class EditModeTestReset
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            var aca = Academy.Instance;

            var numberReset = 0;
            var stepsSinceReset = 0;
            for (var i = 0; i < 50; i++)
            {
                Assert.AreEqual(stepsSinceReset, aca.StepCount);
                Assert.AreEqual(numberReset, aca.EpisodeCount);
                Assert.AreEqual(i, aca.TotalStepCount);
                // Academy resets at the first step
                if (i == 0)
                {
                    numberReset += 1;
                }

                stepsSinceReset += 1;
                aca.EnvironmentStep();
            }
        }

        [Test]
        public void TestAgent()
        {
            var agentGo1 = new GameObject("TestAgent");
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();

            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;

            agent2.LazyInitialize();

            var numberAgent1Reset = 0;
            var numberAgent2Reset = 0;
            var numberAcaReset = 0;
            var acaStepsSinceReset = 0;
            var agent2StepSinceReset = 0;
            for (var i = 0; i < 5000; i++)
            {
                Assert.AreEqual(acaStepsSinceReset, aca.StepCount);
                Assert.AreEqual(numberAcaReset, aca.EpisodeCount);

                Assert.AreEqual(i, aca.TotalStepCount);

                Assert.AreEqual(agent2StepSinceReset, agent2.GetStepCount());
                Assert.AreEqual(numberAgent1Reset, agent1.agentResetCalls);
                Assert.AreEqual(numberAgent2Reset, agent2.agentResetCalls);

                // Agent 2  and academy reset at the first step
                if (i == 0)
                {
                    numberAcaReset += 1;
                    numberAgent2Reset += 1;
                }
                //Agent 1 is only initialized at step 2
                if (i == 2)
                {
                    agent1.LazyInitialize();
                }
                // Set agent 1 to done every 11 steps to test behavior
                if (i % 11 == 5)
                {
                    agent1.Done();
                    numberAgent1Reset += 1;
                }
                // Resetting agent 2 regularly
                if (i % 13 == 3)
                {
                    agent2.Done();
                    numberAgent2Reset += 1;
                    agent2StepSinceReset = 0;
                }
                // Request a decision for agent 2 regularly
                if (i % 3 == 2)
                {
                    agent2.RequestDecision();
                }
                else if (i % 5 == 1)
                {
                    // Request an action without decision regularly
                    agent2.RequestAction();
                }

                acaStepsSinceReset += 1;
                agent2StepSinceReset += 1;
                aca.EnvironmentStep();
            }
        }
    }

    [TestFixture]
    public class EditModeTestMiscellaneous
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestCumulativeReward()
        {
            var agentGo1 = new GameObject("TestAgent");
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();
            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.Awake();


            agent1.maxStep = 20;

            agent2.LazyInitialize();
            agent1.LazyInitialize();
            agent2.SetPolicy(new TestPolicy());

            var j = 0;
            for (var i = 0; i < 500; i++)
            {
                if (i % 21 == 0)
                {
                    j = 0;
                }
                else
                {
                    j++;
                }
                agent2.RequestAction();
                Assert.LessOrEqual(Mathf.Abs(j * 10.1f - agent1.GetCumulativeReward()), 0.05f);
                Assert.LessOrEqual(Mathf.Abs(i * 0.1f - agent2.GetCumulativeReward()), 0.05f);

                agent1.AddReward(10f);
                aca.EnvironmentStep();
            }
        }

        [Test]
        public void TestMaxStepsReset()
        {
            var agentGo1 = new GameObject("TestAgent");
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 1;
            decisionRequester.Awake();

            var maxStep = 6;
            agent1.maxStep = maxStep;
            agent1.LazyInitialize();

            for (var i = 0; i < 15; i++)
            {
                // We expect resets to occur when there are maxSteps actions since the last reset (and on the first step)
                var expectReset = agent1.agentActionCallsSinceLastReset == maxStep || (i == 0);
                var previousNumResets = agent1.agentResetCalls;

                aca.EnvironmentStep();

                if (expectReset)
                {
                    Assert.AreEqual(previousNumResets + 1, agent1.agentResetCalls);
                }
                else
                {
                    Assert.AreEqual(previousNumResets, agent1.agentResetCalls);
                }
            }
        }
    }
}
