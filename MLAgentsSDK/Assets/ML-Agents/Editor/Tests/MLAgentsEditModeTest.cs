using UnityEngine;
using NUnit.Framework;
using System.Reflection;

namespace MLAgents.Tests
{
    public class TestAcademy : Academy
    {
        public int initializeAcademyCalls;
        public int academyResetCalls;
        public int AcademyStepCalls;

        public override void InitializeAcademy()
        {
            initializeAcademyCalls += 1;
        }
        public override void AcademyReset()
        {
            academyResetCalls += 1;
        }

        public override void AcademyStep()
        {
            AcademyStepCalls += 1;
        }
    }
    public class TestAgent : Agent
    {
        public int initializeAgentCalls;
        public int collectObservationsCalls;
        public int agentActionCalls;
        public int agentResetCalls;
        public int agentOnDoneCalls;
        public override void InitializeAgent()
        {
            initializeAgentCalls += 1;
        }
        public override void CollectObservations()
        {
            collectObservationsCalls += 1;
        }

        public override void AgentAction(float[] vectorAction, string textAction)
        {
            agentActionCalls += 1;
            AddReward(0.1f);
        }

        public override void AgentReset()
        {
            agentResetCalls += 1;
        }

        public override void AgentOnDone()
        {
            agentOnDoneCalls += 1;
        }
    }

    // This is an empty class for testing the behavior of agents and academy
    // It is left empty because we are not testing any brain behavior
    public class TestBrain : Brain
    {

    }


    public class EditModeTestGeneration
    {
        [Test]
        public void TestAcademy()
        {
            // Use the Assert class to test conditions.
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            Assert.AreNotEqual(null, aca);
            Assert.AreEqual(0, aca.initializeAcademyCalls);
            Assert.AreEqual(0, aca.GetEpisodeCount());
            Assert.AreEqual(0, aca.GetStepCount());
        }

        [Test]
        public void TestAgent()
        {
            GameObject agentGO = new GameObject("TestAgent");
            agentGO.AddComponent<TestAgent>();
            TestAgent agent = agentGO.GetComponent<TestAgent>();
            Assert.AreNotEqual(null, agent);
            Assert.AreEqual(0, agent.initializeAgentCalls);
        }
    }

    public class EditModeTestInitialization
    {

        [Test]
        public void TestAcademy()
        {
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            Assert.AreEqual(0, aca.initializeAcademyCalls);
            Assert.AreEqual(0, aca.GetStepCount());
            Assert.AreEqual(0, aca.GetEpisodeCount());
            Assert.AreEqual(false, aca.IsDone());
            //This will call the method even though it is private
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("InitializeEnvironment",
                           BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            Assert.AreEqual(1, aca.initializeAcademyCalls);
            Assert.AreEqual(0, aca.GetEpisodeCount());
            Assert.AreEqual(0, aca.GetStepCount());
            Assert.AreEqual(false, aca.IsDone());
            Assert.AreEqual(0, aca.academyResetCalls);
            Assert.AreEqual(0, aca.AcademyStepCalls);
        }

        [Test]
        public void TestAgent()
        {
            GameObject agentGO1 = new GameObject("TestAgent");
            agentGO1.AddComponent<TestAgent>();
            TestAgent agent1 = agentGO1.GetComponent<TestAgent>();
            GameObject agentGO2 = new GameObject("TestAgent");
            agentGO2.AddComponent<TestAgent>();
            TestAgent agent2 = agentGO2.GetComponent<TestAgent>();
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            GameObject brainGO = new GameObject("TestBrain");
            brainGO.transform.parent = acaGO.transform;
            brainGO.AddComponent<TestBrain>();
            TestBrain brain = brainGO.GetComponent<TestBrain>();
            brain.brainParameters = new BrainParameters();
            brain.brainParameters.vectorObservationSize = 0;
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            Assert.AreEqual(false, agent1.IsDone());
            Assert.AreEqual(false, agent2.IsDone());
            Assert.AreEqual(0, agent1.agentResetCalls);
            Assert.AreEqual(0, agent2.agentResetCalls);
            Assert.AreEqual(0, agent1.initializeAgentCalls);
            Assert.AreEqual(0, agent2.initializeAgentCalls);
            Assert.AreEqual(0, agent1.agentActionCalls);
            Assert.AreEqual(0, agent2.agentActionCalls);

            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod("OnEnableHelper",
                   BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("InitializeEnvironment",
                           BindingFlags.Instance | BindingFlags.NonPublic);


            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            AgentEnableMethod.Invoke(agent1, new object[] { aca });

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
        }

    }

    public class EditModeTestStep
    {
        [Test]
        public void TestAcademy()
        {
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("InitializeEnvironment",
                           BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod("EnvironmentStep",
                           BindingFlags.Instance | BindingFlags.NonPublic);

            int numberReset = 0;
            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(numberReset, aca.GetEpisodeCount());
                Assert.AreEqual(i, aca.GetStepCount());
                Assert.AreEqual(false, aca.IsDone());
                Assert.AreEqual(numberReset, aca.academyResetCalls);
                Assert.AreEqual(i, aca.AcademyStepCalls);

                // The reset happens at the begining of the first step
                if (i == 0)
                {
                    numberReset += 1;
                }
                AcademyStepMethod.Invoke((object)aca, new object[] { });
            }
        }

        [Test]
        public void TestAgent()
        {
            GameObject agentGO1 = new GameObject("TestAgent");
            agentGO1.AddComponent<TestAgent>();
            TestAgent agent1 = agentGO1.GetComponent<TestAgent>();
            GameObject agentGO2 = new GameObject("TestAgent");
            agentGO2.AddComponent<TestAgent>();
            TestAgent agent2 = agentGO2.GetComponent<TestAgent>();
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            GameObject brainGO = new GameObject("TestBrain");
            brainGO.transform.parent = acaGO.transform;
            brainGO.AddComponent<TestBrain>();
            TestBrain brain = brainGO.GetComponent<TestBrain>();


            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod(
                "InitializeEnvironment", BindingFlags.Instance | BindingFlags.NonPublic);

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            brain.brainParameters = new BrainParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.onDemandDecision = false;
            agent1.agentParameters.numberOfActionsBetweenDecisions = 2;
            // agent1 will take an action at every step and request a decision every 2 steps
            agent2.agentParameters.onDemandDecision = true;
            // agent2 will request decisions only when RequestDecision is called
            brain.brainParameters.vectorObservationSize = 0;
            brain.brainParameters.cameraResolutions = new resolution[0];
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            AgentEnableMethod.Invoke(agent1, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod(
                "EnvironmentStep", BindingFlags.Instance | BindingFlags.NonPublic);

            int numberAgent1Reset = 0;
            int numberAgent2Initialization = 0;
            int requestDecision = 0;
            int requestAction = 0;
            for (int i = 0; i < 50; i++)
            {
                Assert.AreEqual(numberAgent1Reset, agent1.agentResetCalls);
                // Agent2 is never reset since intialized after academy
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
                    AgentEnableMethod.Invoke(agent2, new object[] { aca });
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
                AcademyStepMethod.Invoke(aca, new object[] { });
            }
        }
    }

    public class EditModeTestReset
    {
        [Test]
        public void TestAcademy()
        {
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod(
                "InitializeEnvironment", BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod(
                "EnvironmentStep", BindingFlags.Instance | BindingFlags.NonPublic);

            int numberReset = 0;
            int stepsSinceReset = 0;
            for (int i = 0; i < 50; i++)
            {

                Assert.AreEqual(stepsSinceReset, aca.GetStepCount());
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(numberReset, aca.GetEpisodeCount());

                Assert.AreEqual(false, aca.IsDone());
                Assert.AreEqual(numberReset, aca.academyResetCalls);
                Assert.AreEqual(i, aca.AcademyStepCalls);
                // Academy resets at the first step
                if (i == 0)
                {
                    numberReset += 1;
                }

                stepsSinceReset += 1;
                // Regularly set the academy to done to check behavior
                if (i % 5 == 3)
                {
                    aca.Done();
                    numberReset += 1;
                    stepsSinceReset = 1;
                    Assert.AreEqual(true, aca.IsDone());
                }
                AcademyStepMethod.Invoke((object)aca, new object[] { });


            }
        }

        [Test]
        public void TestAgent()
        {
            GameObject agentGO1 = new GameObject("TestAgent");
            agentGO1.AddComponent<TestAgent>();
            TestAgent agent1 = agentGO1.GetComponent<TestAgent>();
            GameObject agentGO2 = new GameObject("TestAgent");
            agentGO2.AddComponent<TestAgent>();
            TestAgent agent2 = agentGO2.GetComponent<TestAgent>();
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            GameObject brainGO = new GameObject("TestBrain");
            brainGO.transform.parent = acaGO.transform;
            brainGO.AddComponent<TestBrain>();
            TestBrain brain = brainGO.GetComponent<TestBrain>();


            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod(
                "InitializeEnvironment", BindingFlags.Instance | BindingFlags.NonPublic);

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod(
                "EnvironmentStep", BindingFlags.Instance | BindingFlags.NonPublic);

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            brain.brainParameters = new BrainParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.onDemandDecision = false;
            agent1.agentParameters.numberOfActionsBetweenDecisions = 2;
            // agent1 will take an action at every step and request a decision every 2 steps
            agent2.agentParameters.onDemandDecision = true;
            // agent2 will request decisions only when RequestDecision is called
            brain.brainParameters.vectorObservationSize = 0;
            brain.brainParameters.cameraResolutions = new resolution[0];
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            int numberAgent1Reset = 0;
            int numberAgent2Reset = 0;
            int numberAcaReset = 0;
            int acaStepsSinceReset = 0;
            int agent1StepSinceReset = 0;
            int agent2StepSinceReset = 0;
            int requestDecision = 0;
            int requestAction = 0;
            for (int i = 0; i < 5000; i++)
            {
                Assert.AreEqual(acaStepsSinceReset, aca.GetStepCount());
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(numberAcaReset, aca.GetEpisodeCount());

                Assert.AreEqual(false, aca.IsDone());
                Assert.AreEqual(numberAcaReset, aca.academyResetCalls);
                Assert.AreEqual(i, aca.AcademyStepCalls);

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
                    AgentEnableMethod.Invoke(agent1, new object[] { aca });

                }
                // Reset Academy every 100 steps
                if (i % 100 == 3)
                {
                    aca.Done();
                    numberAcaReset += 1;
                    acaStepsSinceReset = 0;
                }
                // Set agent 1 to done every 11 steps to test behavior
                if (i % 11 == 5)
                {
                    agent1.Done();
                }
                // Reseting agent 2 regularly
                if (i % 13 == 3)
                {
                    if (!(agent2.IsDone() || aca.IsDone()))
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
                    requestDecision += 1;
                    requestAction += 1;
                    agent2.RequestDecision();
                }
                else if (i % 5 == 1)
                {
                    // Request an action without decision regularly
                    requestAction += 1;
                    agent2.RequestAction();
                }
                if (agent1.IsDone() && (((acaStepsSinceReset) % agent1.agentParameters.numberOfActionsBetweenDecisions == 0)) || aca.IsDone())
                {
                    numberAgent1Reset += 1;
                    agent1StepSinceReset = 0;
                }
                if (aca.IsDone())
                {
                    numberAgent2Reset += 1;
                    agent2StepSinceReset = 0;
                }

                acaStepsSinceReset += 1;
                agent1StepSinceReset += 1;
                agent2StepSinceReset += 1;
                //Agent 1 is only initialized at step 2
                if (i < 2)
                {
                    agent1StepSinceReset = 0;
                }
                AcademyStepMethod.Invoke((object)aca, new object[] { });


            }
        }
    }

    public class EditModeTestMaxStep
    {
        [Test]
        public void TestAcademy()
        {
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod(
                "InitializeEnvironment", BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod(
                "EnvironmentStep", BindingFlags.Instance | BindingFlags.NonPublic);

            FieldInfo maxStep = typeof(Academy).GetField(
                "maxSteps", BindingFlags.Instance | BindingFlags.NonPublic);
            maxStep.SetValue((object)aca, 20);

            int numberReset = 0;
            int stepsSinceReset = 0;
            for (int i = 0; i < 50; i++)
            {
                Assert.AreEqual(stepsSinceReset, aca.GetStepCount());
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(false, aca.IsDone());

                Assert.AreEqual(i, aca.AcademyStepCalls);
                Assert.AreEqual(numberReset, aca.GetEpisodeCount());
                Assert.AreEqual(numberReset, aca.academyResetCalls);
                stepsSinceReset += 1;
                // Make sure max step is reached every 20 steps
                if (i % 20 == 0)
                {
                    numberReset += 1;
                    stepsSinceReset = 1;

                }
                AcademyStepMethod.Invoke((object)aca, new object[] { });
            }
        }

        [Test]
        public void TestAgent()
        {
            GameObject agentGO1 = new GameObject("TestAgent");
            agentGO1.AddComponent<TestAgent>();
            TestAgent agent1 = agentGO1.GetComponent<TestAgent>();
            GameObject agentGO2 = new GameObject("TestAgent");
            agentGO2.AddComponent<TestAgent>();
            TestAgent agent2 = agentGO2.GetComponent<TestAgent>();
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            GameObject brainGO = new GameObject("TestBrain");
            brainGO.transform.parent = acaGO.transform;
            brainGO.AddComponent<TestBrain>();
            TestBrain brain = brainGO.GetComponent<TestBrain>();


            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod(
                "InitializeEnvironment", BindingFlags.Instance | BindingFlags.NonPublic);

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod(
                "EnvironmentStep", BindingFlags.Instance | BindingFlags.NonPublic);

            FieldInfo maxStep = typeof(Academy).GetField(
                "maxSteps", BindingFlags.Instance | BindingFlags.NonPublic);
            maxStep.SetValue((object)aca, 100);

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            brain.brainParameters = new BrainParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.onDemandDecision = false;
            agent1.agentParameters.numberOfActionsBetweenDecisions = 1;
            // agent1 will take an action at every step and request a decision every 2 steps
            agent2.agentParameters.onDemandDecision = true;
            // agent2 will request decisions only when RequestDecision is called
            agent1.agentParameters.maxStep = 20;
            agent2.agentParameters.maxStep = 30;
            brain.brainParameters.vectorObservationSize = 0;
            brain.brainParameters.cameraResolutions = new resolution[0];
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });


            int numberAgent1Reset = 0;
            int numberAgent2Reset = 0;
            int numberAcaReset = 0;
            int acaStepsSinceReset = 0;
            int agent1StepSinceReset = 0;
            int agent2StepSinceReset = 0;

            for (int i = 0; i < 500; i++)
            {
                Assert.AreEqual(acaStepsSinceReset, aca.GetStepCount());
                Assert.AreEqual(1, aca.initializeAcademyCalls);

                Assert.AreEqual(i, aca.AcademyStepCalls);

                Assert.AreEqual(agent1StepSinceReset, agent1.GetStepCount());
                Assert.AreEqual(agent2StepSinceReset, agent2.GetStepCount());


                Assert.AreEqual(numberAcaReset, aca.GetEpisodeCount());
                Assert.AreEqual(numberAcaReset, aca.academyResetCalls);
                Assert.AreEqual(numberAgent1Reset, agent1.agentResetCalls);
                Assert.AreEqual(numberAgent2Reset, agent2.agentResetCalls);

                //At the first step, Academy and agent 2 reset
                if (i == 0)
                {
                    numberAcaReset += 1;
                    numberAgent2Reset += 1;
                }
                //Agent 1 is only initialized at step 2
                if (i == 2)
                {
                    AgentEnableMethod.Invoke(agent1, new object[] { aca });
                }

                // we request a decision at each step
                agent2.RequestDecision();

                if (i > 3)
                {
                    // Make sure the academy max steps at 100
                    if (i % 100 == 0)
                    {
                        acaStepsSinceReset = 0;
                        agent1StepSinceReset = 0;
                        agent2StepSinceReset = 0;
                        numberAcaReset += 1;
                        numberAgent1Reset += 1;
                        numberAgent2Reset += 1;
                    }
                    else
                    {
                        //Make sure the agents reset when their max steps is reached
                        if (agent1StepSinceReset % 21 == 0)
                        {
                            agent1StepSinceReset = 0;
                            numberAgent1Reset += 1;
                        }
                        if (agent2StepSinceReset % 31 == 0)
                        {
                            agent2StepSinceReset = 0;
                            numberAgent2Reset += 1;
                        }
                    }
                }

                acaStepsSinceReset += 1;
                agent1StepSinceReset += 1;
                agent2StepSinceReset += 1;

                //Agent 1 is only initialized at step 2
                if (i < 2)
                {
                    agent1StepSinceReset = 0;
                }

                AcademyStepMethod.Invoke((object)aca, new object[] { });

            }

        }
    }

    public class EditModeTestMiscellaneous
    {
        [Test]
        public void TestResetOnDone()
        {
            GameObject agentGO1 = new GameObject("TestAgent");
            agentGO1.AddComponent<TestAgent>();
            TestAgent agent1 = agentGO1.GetComponent<TestAgent>();
            GameObject agentGO2 = new GameObject("TestAgent");
            agentGO2.AddComponent<TestAgent>();
            TestAgent agent2 = agentGO2.GetComponent<TestAgent>();
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            GameObject brainGO = new GameObject("TestBrain");
            brainGO.transform.parent = acaGO.transform;
            brainGO.AddComponent<TestBrain>();
            TestBrain brain = brainGO.GetComponent<TestBrain>();


            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod(
                "InitializeEnvironment", BindingFlags.Instance | BindingFlags.NonPublic);

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod(
                "EnvironmentStep", BindingFlags.Instance | BindingFlags.NonPublic);

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            brain.brainParameters = new BrainParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.onDemandDecision = false;
            // agent1 will take an action at every step and request a decision every steps
            agent1.agentParameters.numberOfActionsBetweenDecisions = 1;
            // agent2 will request decisions only when RequestDecision is called
            agent2.agentParameters.onDemandDecision = true;
            agent1.agentParameters.maxStep = 20;
            //Here we specify that the agent does not reset when done
            agent1.agentParameters.resetOnDone = false;
            agent2.agentParameters.resetOnDone = false;
            brain.brainParameters.vectorObservationSize = 0;
            brain.brainParameters.cameraResolutions = new resolution[0];
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            AgentEnableMethod.Invoke(agent1, new object[] { aca });

            int agent1ResetOnDone = 0;
            int agent2ResetOnDone = 0;
            int acaStepsSinceReset = 0;
            int agent1StepSinceReset = 0;
            int agent2StepSinceReset = 0;

            for (int i = 0; i < 50; i++)
            {
                Assert.AreEqual(i, aca.AcademyStepCalls);

                Assert.AreEqual(agent1StepSinceReset, agent1.GetStepCount());
                Assert.AreEqual(agent2StepSinceReset, agent2.GetStepCount());
                Assert.AreEqual(agent1ResetOnDone, agent1.agentOnDoneCalls);
                Assert.AreEqual(agent2ResetOnDone, agent2.agentOnDoneCalls);

                // we request a decision at each step
                agent2.RequestDecision();
                acaStepsSinceReset += 1;
                if (agent1ResetOnDone == 0)
                    agent1StepSinceReset += 1;
                if (agent2ResetOnDone == 0)
                    agent2StepSinceReset += 1;

                if ((i > 2) && (i % 21 == 0))
                {
                    agent1ResetOnDone = 1;
                }

                if (i == 31)
                {
                    agent2ResetOnDone = 1;
                    agent2.Done();
                }


                AcademyStepMethod.Invoke((object)aca, new object[] { });
            }

        }

        [Test]
        public void TestCumulativeReward()
        {
            GameObject agentGO1 = new GameObject("TestAgent");
            agentGO1.AddComponent<TestAgent>();
            TestAgent agent1 = agentGO1.GetComponent<TestAgent>();
            GameObject agentGO2 = new GameObject("TestAgent");
            agentGO2.AddComponent<TestAgent>();
            TestAgent agent2 = agentGO2.GetComponent<TestAgent>();
            GameObject acaGO = new GameObject("TestAcademy");
            acaGO.AddComponent<TestAcademy>();
            TestAcademy aca = acaGO.GetComponent<TestAcademy>();
            GameObject brainGO = new GameObject("TestBrain");
            brainGO.transform.parent = acaGO.transform;
            brainGO.AddComponent<TestBrain>();
            TestBrain brain = brainGO.GetComponent<TestBrain>();


            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod(
                "OnEnableHelper", BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod(
                "InitializeEnvironment", BindingFlags.Instance | BindingFlags.NonPublic);

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod(
                "EnvironmentStep", BindingFlags.Instance | BindingFlags.NonPublic);

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            brain.brainParameters = new BrainParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.onDemandDecision = false;
            agent1.agentParameters.numberOfActionsBetweenDecisions = 3;
            // agent1 will take an action at every step and request a decision every 2 steps
            agent2.agentParameters.onDemandDecision = true;
            // agent2 will request decisions only when RequestDecision is called
            agent1.agentParameters.maxStep = 20;
            brain.brainParameters.vectorObservationSize = 0;
            brain.brainParameters.cameraResolutions = new resolution[0];
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            AgentEnableMethod.Invoke(agent1, new object[] { aca });


            int j = 0;
            for (int i = 0; i < 500; i++)
            {
                agent2.RequestAction();
                Assert.LessOrEqual(Mathf.Abs(j * 0.1f + j * 10f - agent1.GetCumulativeReward()), 0.05f);
                Assert.LessOrEqual(Mathf.Abs(i * 0.1f - agent2.GetCumulativeReward()), 0.05f);


                AcademyStepMethod.Invoke((object)aca, new object[] { });
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
