using UnityEngine;
using NUnit.Framework;
using System.Reflection;
using MLAgents.Sensor;

namespace MLAgents.Tests
{
    public class TestAgent : Agent
    {
        public int initializeAgentCalls;
        public int collectObservationsCalls;
        public int agentActionCalls;
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

        public override void CollectObservations()
        {
            collectObservationsCalls += 1;
            AddVectorObs(0f);
        }

        public override void AgentAction(float[] vectorAction)
        {
            agentActionCalls += 1;
            AddReward(0.1f);
        }

        public override void AgentReset()
        {
            agentResetCalls += 1;
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

        public void Update() { }
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
            Assert.AreEqual(0, aca.GetEpisodeCount());
            Assert.AreEqual(0, aca.GetStepCount());
            Assert.AreEqual(0, aca.GetTotalStepCount());
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
            aca.LazyInitialization();
            aca.LazyInitialization();

            Assert.AreEqual(0, aca.GetEpisodeCount());
            Assert.AreEqual(0, aca.GetStepCount());
            Assert.AreEqual(0, aca.GetTotalStepCount());
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

            Assert.AreEqual(false, agent1.IsDone());
            Assert.AreEqual(false, agent2.IsDone());
            Assert.AreEqual(0, agent1.agentResetCalls);
            Assert.AreEqual(0, agent2.agentResetCalls);
            Assert.AreEqual(0, agent1.initializeAgentCalls);
            Assert.AreEqual(0, agent2.initializeAgentCalls);
            Assert.AreEqual(0, agent1.agentActionCalls);
            Assert.AreEqual(0, agent2.agentActionCalls);

            var agentEnableMethod = typeof(Agent).GetMethod("OnEnableHelper",
                BindingFlags.Instance | BindingFlags.NonPublic);

            agentEnableMethod?.Invoke(agent2, new object[] { });
            agentEnableMethod?.Invoke(agent1, new object[] { });

            Assert.AreEqual(false, agent1.IsDone());
            Assert.AreEqual(false, agent2.IsDone());
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
                Assert.AreEqual(numberReset, aca.GetEpisodeCount());
                Assert.AreEqual(i, aca.GetStepCount());

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
            Assert.IsTrue(aca.IsAutomaticSteppingEnabled);
            aca.DisableAutomaticStepping(true);
            Assert.IsFalse(aca.IsAutomaticSteppingEnabled);
            aca.EnableAutomaticStepping();
            Assert.IsTrue(aca.IsAutomaticSteppingEnabled);
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

            var agentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.Awake();
            // agent1 will take an action at every step and request a decision every 2 steps
            // agent2 will request decisions only when RequestDecision is called

            agentEnableMethod?.Invoke(agent1, new object[] { });

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
                    agentEnableMethod?.Invoke(agent2, new object[] { });
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
                Assert.AreEqual(stepsSinceReset, aca.GetStepCount());
                Assert.AreEqual(numberReset, aca.GetEpisodeCount());
                Assert.AreEqual(i, aca.GetTotalStepCount());
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

            var agentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;

            agentEnableMethod?.Invoke(agent2, new object[] { });

            var numberAgent1Reset = 0;
            var numberAgent2Reset = 0;
            var numberAcaReset = 0;
            var acaStepsSinceReset = 0;
            var agent2StepSinceReset = 0;
            for (var i = 0; i < 5000; i++)
            {
                Assert.AreEqual(acaStepsSinceReset, aca.GetStepCount());
                Assert.AreEqual(numberAcaReset, aca.GetEpisodeCount());

                Assert.AreEqual(i, aca.GetTotalStepCount());

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
                    agentEnableMethod?.Invoke(agent1, new object[] { });
                }
                // Set agent 1 to done every 11 steps to test behavior
                if (i % 11 == 5)
                {
                    agent1.Done();
                }
                // Resetting agent 2 regularly
                if (i % 13 == 3)
                {
                    if (!(agent2.IsDone()))
                    {
                        // If the agent was already reset before the request decision
                        // We should not reset again
                        agent2.Done();
                        numberAgent2Reset += 1;
                        agent2StepSinceReset = 0;
                    }
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
                if (agent1.IsDone())
                {
                    numberAgent1Reset += 1;
                }

                acaStepsSinceReset += 1;
                agent2StepSinceReset += 1;
                //Agent 1 is only initialized at step 2
                if (i < 2)
                { }
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

            var agentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);
            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.Awake();

            agent1.agentParameters.maxStep = 20;

            agentEnableMethod?.Invoke(agent2, new object[] { });
            agentEnableMethod?.Invoke(agent1, new object[] { });


            var j = 0;
            for (var i = 0; i < 500; i++)
            {
                agent2.RequestAction();
                Assert.LessOrEqual(Mathf.Abs(j * 0.1f + j * 10f - agent1.GetCumulativeReward()), 0.05f);
                Assert.LessOrEqual(Mathf.Abs(i * 0.1f - agent2.GetCumulativeReward()), 0.05f);


                aca.EnvironmentStep();
                agent1.AddReward(10f);

                if ((i % 21 == 0) && (i > 0))
                {
                    j = 0;
                }
                j++;
            }
        }
    }
}
