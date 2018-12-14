using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using System.Reflection;
using MLAgents.InferenceBrain;

namespace MLAgents.Tests
{
    public class EditModeTestInternalBrainTensorApplier
    {
        private class TestAgent : Agent
        {
            public AgentAction GetAction()
            {
                FieldInfo f =  typeof(Agent).GetField(
                    "action", BindingFlags.Instance | BindingFlags.NonPublic);
                return (AgentAction) f.GetValue(this);
            }
        }
        
        private Dictionary<Agent, AgentInfo> GetFakeAgentInfos()
        {
            var goA = new GameObject("goA");
            var agentA = goA.AddComponent<TestAgent>();
            var infoA = new AgentInfo();
            var goB = new GameObject("goB");
            var agentB = goB.AddComponent<TestAgent>();
            var infoB = new AgentInfo();

            return new Dictionary<Agent, AgentInfo>(){{agentA, infoA},{agentB, infoB}};
        }

        [Test]
        public void Contruction()
        {
            var bp = new BrainParameters();
            var tensorGenerator = new TensorApplier(bp, 0);
            Assert.IsNotNull(tensorGenerator);
        }

        [Test]
        public void ApplyContinuousActionOutput()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 3},
                Data = new float[,] {{1, 2, 3}, {4, 5, 6}}
            };
            var agentInfos = GetFakeAgentInfos();
            
            var applier = new ContinuousActionOutputApplier();
            applier.Apply(inputTensor, agentInfos);
            var agents = agentInfos.Keys.ToList();
            var agent = agents[0] as TestAgent;
            var action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 1);
            Assert.AreEqual(action.vectorActions[1], 2);
            Assert.AreEqual(action.vectorActions[2], 3);
            agent = agents[1] as TestAgent;
            action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 4);
            Assert.AreEqual(action.vectorActions[1], 5);
            Assert.AreEqual(action.vectorActions[2], 6); 
        }
        
        [Test]
        public void ApplyDiscreteActionOutput()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 5},
                Data = new float[,] {{0.5f, 22.5f, 0.1f, 5f, 1f},
                    {4f, 5f, 6f, 7f, 8f}}
            };
            var agentInfos = GetFakeAgentInfos();
            
            var applier = new DiscreteActionOutputApplier(new int[]{2, 3}, 0);
            applier.Apply(inputTensor, agentInfos);
            var agents = agentInfos.Keys.ToList();
            var agent = agents[0] as TestAgent;
            var action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 1);
            Assert.AreEqual(action.vectorActions[1], 1);
            agent = agents[1] as TestAgent;
            action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 1);
            Assert.AreEqual(action.vectorActions[1], 2);
        }
        
        [Test]
        public void ApplyMemoryOutput()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 5},
                Data = new float[,] {{0.5f, 22.5f, 0.1f, 5f, 1f},
                    {4f, 5f, 6f, 7f, 8f}}
            };
            var agentInfos = GetFakeAgentInfos();
            
            var applier = new MemoryOutputApplier();
            applier.Apply(inputTensor, agentInfos);
            var agents = agentInfos.Keys.ToList();
            var agent = agents[0] as TestAgent;
            var action = agent.GetAction();
            Assert.AreEqual(action.memories[0], 0.5f);
            Assert.AreEqual(action.memories[1], 22.5f);
            agent = agents[1] as TestAgent;
            action = agent.GetAction();
            Assert.AreEqual(action.memories[2], 6);
            Assert.AreEqual(action.memories[3], 7);
        }
        
        [Test]
        public void ApplyValueEstimate()
        {
            var inputTensor = new Tensor()
            {
                Shape = new long[] {2, 1},
                Data = new float[,] {{0.5f}, {8f}}
            };
            var agentInfos = GetFakeAgentInfos();
            
            var applier = new ValueEstimateApplier();
            applier.Apply(inputTensor, agentInfos);
            var agents = agentInfos.Keys.ToList();
            var agent = agents[0] as TestAgent;
            var action = agent.GetAction();
            Assert.AreEqual(action.value, 0.5f);
            agent = agents[1] as TestAgent;
            action = agent.GetAction();
            Assert.AreEqual(action.value, 8);
        }
    }
}
