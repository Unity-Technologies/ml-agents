using UnityEngine;
using UnityEditor;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;
using System.Reflection;

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



public class MLAgentsEditModeTestGeneration
{
    [Test]
    public void TestAcademy()
    {
        // Use the Assert class to test conditions.
        GameObject acaGO = new GameObject("TestAcademy");
        acaGO.AddComponent<TestAcademy>();
        TestAcademy aca = acaGO.GetComponent<TestAcademy>();
        Assert.AreNotEqual(aca, null);
        Assert.AreEqual(aca.initializeAcademyCalls, 0);
        Assert.AreEqual(aca.episodeCount, 0);
        Assert.AreEqual(aca.stepsSinceReset, 0);
    }

    [Test]
    public void TestAgent()
    {
        GameObject agentGO = new GameObject("TestAgent");
        agentGO.AddComponent<TestAgent>();
        TestAgent agent = agentGO.GetComponent<TestAgent>();
        Assert.AreNotEqual(agent, null);
        Assert.AreEqual(agent.initializeAgentCalls, 0);
    }
}

public class MLAgentsEditModeTestInitialization
{

    [Test]
    public void TestAcademy()
    {
        GameObject acaGO = new GameObject("TestAcademy");
        acaGO.AddComponent<TestAcademy>();
        TestAcademy aca = acaGO.GetComponent<TestAcademy>();
        Assert.AreEqual(aca.initializeAcademyCalls, 0);
        Assert.AreEqual(aca.stepsSinceReset, 0);
        Assert.AreEqual(aca.episodeCount, 0);
        Assert.AreEqual(aca.IsDone(), false);
        //This will call the method even though it is private
        MethodInfo privMethod = typeof(Academy).GetMethod("_InitializeAcademy", BindingFlags.Instance | BindingFlags.NonPublic);
        privMethod.Invoke(aca, new object[] { });
        Assert.AreEqual(1, aca.initializeAcademyCalls);
        Assert.AreEqual(aca.episodeCount, 0);
        Assert.AreEqual(aca.stepsSinceReset, 0);
        Assert.AreEqual(aca.IsDone(), false);
    }

}
