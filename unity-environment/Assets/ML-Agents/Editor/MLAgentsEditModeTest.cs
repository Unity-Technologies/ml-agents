using UnityEngine;
using UnityEditor;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;
using System.Reflection;

namespace MLAgentsTests
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

        public override void AgentAction(float[] act)
        {
            agentActionCalls += 1;
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

    public class TestBrain : Brain
    {
        // TODO : Mock a brain
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
            Assert.AreEqual(0, aca.episodeCount);
            Assert.AreEqual(0, aca.stepsSinceReset);
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
            Assert.AreEqual(0, aca.stepsSinceReset);
            Assert.AreEqual(0, aca.episodeCount);
            Assert.AreEqual(false, aca.IsDone());
            //This will call the method even though it is private
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("_InitializeAcademy",
                           BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            Assert.AreEqual(1, aca.initializeAcademyCalls);
            Assert.AreEqual(1, aca.episodeCount);
            Assert.AreEqual(0, aca.stepsSinceReset);
            Assert.AreEqual(false, aca.IsDone());
            Assert.AreEqual(1, aca.academyResetCalls);
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
            brain.brainParameters.stateSize = 0;
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

            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod("_InitializeAgent",
                   BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("_InitializeAcademy",
                           BindingFlags.Instance | BindingFlags.NonPublic);


            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            AgentEnableMethod.Invoke(agent1, new object[] { aca });

            Assert.AreEqual(false, agent1.IsDone());
            Assert.AreEqual(false, agent2.IsDone());
            //agent1 was not enabled when the academy started
            Assert.AreEqual(0, agent1.agentResetCalls);
            Assert.AreEqual(1, agent2.agentResetCalls);
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
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("_InitializeAcademy",
                           BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod("_AcademyStep",
                           BindingFlags.Instance | BindingFlags.NonPublic);

            for (int i = 0; i < 10; i++){  
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(1, aca.episodeCount);
                Assert.AreEqual(i, aca.stepsSinceReset);
                Assert.AreEqual(false, aca.IsDone());
                Assert.AreEqual(1, aca.academyResetCalls);
                Assert.AreEqual(i, aca.AcademyStepCalls);

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


            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod("_InitializeAgent",
                   BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("_InitializeAcademy",
                           BindingFlags.Instance | BindingFlags.NonPublic);




            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            brain.brainParameters = new BrainParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.eventBased = false;
            agent1.agentParameters.numberOfActionsBetweenDecisions = 2;
            // agent1 will take an action at every step and request a decision every 2 steps
            agent2.agentParameters.eventBased = true;
            // agent2 will request decisions only when RequestDecision is called
            brain.brainParameters.stateSize = 0;
            brain.brainParameters.cameraResolutions = new resolution[0];
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            AgentEnableMethod.Invoke(agent1, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            AgentEnableMethod.Invoke(agent2, new object[] { aca });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod("_AcademyStep",
                           BindingFlags.Instance | BindingFlags.NonPublic);

            int requestDecision =0;
            int requestAction=0;
            for (int i = 0; i < 50; i++)
            {
                
                Assert.AreEqual(1, agent1.agentResetCalls);
                Assert.AreEqual(0, agent2.agentResetCalls);
                Assert.AreEqual(1, agent1.initializeAgentCalls);
                Assert.AreEqual(1, agent2.initializeAgentCalls);
                Assert.AreEqual(i, agent1.agentActionCalls);
                Assert.AreEqual(requestAction, agent2.agentActionCalls);
                Assert.AreEqual((i+1)/2, agent1.collectObservationsCalls);
                Assert.AreEqual(requestDecision, agent2.collectObservationsCalls);
                if (i % 3 == 0)
                {
                    requestDecision +=1;
                    requestAction+=1;
                    agent2.RequestDecision();
                }
                else if (i % 5 == 0)
                {
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
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("_InitializeAcademy",
                           BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod("_AcademyStep",
                           BindingFlags.Instance | BindingFlags.NonPublic);

            int numberReset = 1;
            int stepsSinceReset = 0;
            for (int i = 0; i < 50; i++)
            {

                Assert.AreEqual(stepsSinceReset, aca.stepsSinceReset);
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(numberReset, aca.episodeCount);

                Assert.AreEqual(false, aca.IsDone());
                Assert.AreEqual(numberReset, aca.academyResetCalls);
                Assert.AreEqual(i, aca.AcademyStepCalls);

                stepsSinceReset += 1;
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


            MethodInfo AgentEnableMethod = typeof(Agent).GetMethod("_InitializeAgent",
                   BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("_InitializeAcademy",
                           BindingFlags.Instance | BindingFlags.NonPublic);

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod("_AcademyStep",
                           BindingFlags.Instance | BindingFlags.NonPublic);




            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            brain.brainParameters = new BrainParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.eventBased = false;
            agent1.agentParameters.numberOfActionsBetweenDecisions = 2;
            // agent1 will take an action at every step and request a decision every 2 steps
            agent2.agentParameters.eventBased = true;
            // agent2 will request decisions only when RequestDecision is called
            brain.brainParameters.stateSize = 0;
            brain.brainParameters.cameraResolutions = new resolution[0];
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            AgentEnableMethod.Invoke(agent1, new object[] { aca });

            int numberAgent1Reset = 0; // Agent1 was not enabled at Academy start
            int numberAgent2Reset = 1; 
            int numberAcaReset = 1;
            int acaStepsSinceReset = 0;
            int agent1StepSinceReset =0;
            int agent2StepSinceReset=0;
            int requestDecision = 0;
            int requestAction = 0;
            for (int i = 0; i < 5000; i++)
            {
                Assert.AreEqual(acaStepsSinceReset, aca.stepsSinceReset);
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(numberAcaReset, aca.episodeCount);

                Assert.AreEqual(false, aca.IsDone());
                Assert.AreEqual(numberAcaReset, aca.academyResetCalls);
                Assert.AreEqual(i, aca.AcademyStepCalls);

                Assert.AreEqual(agent1StepSinceReset, agent1.stepCounter);
                Assert.AreEqual(agent2StepSinceReset, agent2.stepCounter);

                acaStepsSinceReset += 1;
                agent1StepSinceReset += 1;
                agent2StepSinceReset += 1;

                if (i % 100 == 3)
                {
                    aca.Done();
                    numberAcaReset += 1;
                    acaStepsSinceReset = 1;
                }
                if (i % 11 == 5)
                {
                    agent1.Done();
                }
                if (i % 13 == 3)
                {
                    if (!agent2.IsDone())
                    {
                        // If the agent was already reset before the request decision
                        // We should not reset again
                        agent2.Done();
                        numberAgent2Reset += 1;
                        agent2StepSinceReset = 1;
                    }
                }

                if (i % 3 == 2)
                {
                    requestDecision += 1;
                    requestAction += 1;
                    agent2.RequestDecision();
                }
                else if (i % 5 == 1)
                {
                    requestAction += 1;
                    agent2.RequestAction();
                }
                if (agent1.IsDone() && (((acaStepsSinceReset+1) % agent1.agentParameters.numberOfActionsBetweenDecisions==0)) || aca.IsDone())
                {
                    numberAgent1Reset += 1;
                    agent1StepSinceReset = 1;
                }
                if (aca.IsDone())
                {
                    numberAgent2Reset += 1;
                    agent2StepSinceReset = 1;
                }

                AcademyStepMethod.Invoke((object)aca, new object[] { });


            }
        }
    }

    //public class EditModeTestMaxStep
    //{
    //    [Test]
    //    public void TestAcademy()
    //    {
    //        //TODO
    //        Assert.AreEqual(0, 1);
    //    }

    //    [Test]
    //    public void TestAgent()
    //    {
    //        //TODO
    //        Assert.AreEqual(0, 1);
    //    }
    //}

    //public class EditModeTestResetOnDone
    //{
    //    [Test]
    //    public void TestAcademy()
    //    {
    //        //TODO
    //        Assert.AreEqual(0, 1);
    //    }

    //    [Test]
    //    public void TestAgent()
    //    {
    //        //TODO
    //        Assert.AreEqual(0, 1);
    //    }
    //}
}
