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
            Assert.AreEqual(0, aca.episodeCount);
            Assert.AreEqual(0, aca.stepsSinceReset);
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
            MethodInfo AcademyInitializeMethod = typeof(Academy).GetMethod("_InitializeAcademy",
                           BindingFlags.Instance | BindingFlags.NonPublic);
            AcademyInitializeMethod.Invoke(aca, new object[] { });

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod("_AcademyStep",
                           BindingFlags.Instance | BindingFlags.NonPublic);

            for (int i = 0; i < 10; i++){
                Assert.AreEqual(1, aca.initializeAcademyCalls);
                Assert.AreEqual(0, aca.episodeCount);
                Assert.AreEqual(i, aca.stepsSinceReset);
                Assert.AreEqual(false, aca.IsDone());
                Assert.AreEqual(0, aca.academyResetCalls);
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


            AgentEnableMethod.Invoke(agent2, new object[] { aca });
            AcademyInitializeMethod.Invoke(aca, new object[] { });
            AgentEnableMethod.Invoke(agent1, new object[] { aca });

            agent1.agentParameters = new AgentParameters();
            agent2.agentParameters = new AgentParameters();
            // We use event based so the agent will now try to send anything to the brain
            agent1.agentParameters.eventBased = true;
            agent2.agentParameters.eventBased = true;
            agent1.GiveBrain(brain);
            agent2.GiveBrain(brain);

            MethodInfo AcademyStepMethod = typeof(Academy).GetMethod("_AcademyStep",
                           BindingFlags.Instance | BindingFlags.NonPublic);

            for (int i = 0; i < 10; i++)
            {
                Debug.Log(agent1.agentActionCalls);
                Assert.AreEqual(0, agent1.agentResetCalls);
                Assert.AreEqual(0, agent2.agentResetCalls);
                Assert.AreEqual(1, agent1.initializeAgentCalls);
                Assert.AreEqual(1, agent2.initializeAgentCalls);
                Assert.AreEqual(0, agent1.agentActionCalls);
                Assert.AreEqual(0, agent2.agentActionCalls);
                Assert.AreEqual(0, agent1.collectObservationsCalls);
                Assert.AreEqual(0, agent2.collectObservationsCalls);
                AcademyStepMethod.Invoke(aca, new object[] { });

            }
        }
    }

    public class EditModeTestReset
    {
        [Test]
        public void TestAcademy()
        {
            //TODO
            Assert.AreEqual(0, 1);
        }

        [Test]
        public void TestAgent()
        {
            //TODO
            Assert.AreEqual(0, 1);
        }
    }

    public class EditModeTestResetOnDone
    {
        [Test]
        public void TestAcademy()
        {
            //TODO
            Assert.AreEqual(0, 1);
        }

        [Test]
        public void TestAgent()
        {
            //TODO
            Assert.AreEqual(0, 1);
        }
    }
}
