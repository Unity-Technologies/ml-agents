using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using System.Reflection;
using Barracuda;
using MLAgents.InferenceBrain;

namespace MLAgents.Tests
{
    public class EditModeTestInternalBrainTensorApplier
    {
        class TestAgent : Agent
        {
            public AgentAction GetAction()
            {
                var f = typeof(Agent).GetField(
                    "m_Action", BindingFlags.Instance | BindingFlags.NonPublic);
                return (AgentAction)f.GetValue(this);
            }
        }

        List<Agent> GetFakeAgentInfos()
        {
            var goA = new GameObject("goA");
            var agentA = goA.AddComponent<TestAgent>();
            var goB = new GameObject("goB");
            var agentB = goB.AddComponent<TestAgent>();

            return new List<Agent> { agentA, agentB };
        }

        [Test]
        public void Construction()
        {
            var bp = new BrainParameters();
            var alloc = new TensorCachingAllocator();
            var mem = new Dictionary<int, List<float>>();
            var tensorGenerator = new TensorApplier(bp, 0, alloc, mem);
            Assert.IsNotNull(tensorGenerator);
            alloc.Dispose();
        }

        [Test]
        public void ApplyContinuousActionOutput()
        {
            var inputTensor = new TensorProxy()
            {
                shape = new long[] { 2, 3 },
                data = new Tensor(2, 3, new float[] { 1, 2, 3, 4, 5, 6 })
            };
            var agentInfos = GetFakeAgentInfos();

            var applier = new ContinuousActionOutputApplier();
            applier.Apply(inputTensor, agentInfos);
            var agents = agentInfos;

            var agent = agents[0] as TestAgent;
            Assert.NotNull(agent);
            var action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 1);
            Assert.AreEqual(action.vectorActions[1], 2);
            Assert.AreEqual(action.vectorActions[2], 3);

            agent = agents[1] as TestAgent;
            Assert.NotNull(agent);
            action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 4);
            Assert.AreEqual(action.vectorActions[1], 5);
            Assert.AreEqual(action.vectorActions[2], 6);
        }

        [Test]
        public void ApplyDiscreteActionOutput()
        {
            var inputTensor = new TensorProxy()
            {
                shape = new long[] { 2, 5 },
                data = new Tensor(
                    2,
                    5,
                    new[] { 0.5f, 22.5f, 0.1f, 5f, 1f, 4f, 5f, 6f, 7f, 8f })
            };
            var agentInfos = GetFakeAgentInfos();
            var alloc = new TensorCachingAllocator();
            var applier = new DiscreteActionOutputApplier(new[] { 2, 3 }, 0, alloc);
            applier.Apply(inputTensor, agentInfos);
            var agents = agentInfos;

            var agent = agents[0] as TestAgent;
            Assert.NotNull(agent);
            var action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 1);
            Assert.AreEqual(action.vectorActions[1], 1);

            agent = agents[1] as TestAgent;
            Assert.NotNull(agent);
            action = agent.GetAction();
            Assert.AreEqual(action.vectorActions[0], 1);
            Assert.AreEqual(action.vectorActions[1], 2);
            alloc.Dispose();
        }

        [Test]
        public void ApplyValueEstimate()
        {
            var inputTensor = new TensorProxy()
            {
                shape = new long[] { 2, 1 },
                data = new Tensor(2, 1, new[] { 0.5f, 8f })
            };
            var agentInfos = GetFakeAgentInfos();

            var applier = new ValueEstimateApplier();
            applier.Apply(inputTensor, agentInfos);
            var agents = agentInfos;

            var agent = agents[0] as TestAgent;
            Assert.NotNull(agent);
            var action = agent.GetAction();
            Assert.AreEqual(action.value, 0.5f);

            agent = agents[1] as TestAgent;
            Assert.NotNull(agent);
            action = agent.GetAction();
            Assert.AreEqual(action.value, 8);
        }
    }
}
