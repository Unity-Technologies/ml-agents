using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using System.Reflection;
using Barracuda;
using MLAgents.InferenceBrain;
using System;

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

            var applier = new ContinuousActionOutputApplier();

            var action0 = new AgentAction();
            var action1 = new AgentAction();
            var callbacks = new List<AgentIdActionPair>()
            {
                new AgentIdActionPair{agentId = 0, action = (a) => action0 = a},
                new AgentIdActionPair{agentId = 1, action = (a) => action1 = a}
            };

            applier.Apply(inputTensor, callbacks);


            Assert.AreEqual(action0.vectorActions[0], 1);
            Assert.AreEqual(action0.vectorActions[1], 2);
            Assert.AreEqual(action0.vectorActions[2], 3);

            Assert.AreEqual(action1.vectorActions[0], 4);
            Assert.AreEqual(action1.vectorActions[1], 5);
            Assert.AreEqual(action1.vectorActions[2], 6);
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
            var alloc = new TensorCachingAllocator();
            var applier = new DiscreteActionOutputApplier(new[] { 2, 3 }, 0, alloc);

            var action0 = new AgentAction();
            var action1 = new AgentAction();
            var callbacks = new List<AgentIdActionPair>()
            {
                new AgentIdActionPair{agentId = 0, action = (a) => action0 = a},
                new AgentIdActionPair{agentId = 1, action = (a) => action1 = a}
            };

            applier.Apply(inputTensor, callbacks);

            Assert.AreEqual(action0.vectorActions[0], 1);
            Assert.AreEqual(action0.vectorActions[1], 1);

            Assert.AreEqual(action1.vectorActions[0], 1);
            Assert.AreEqual(action1.vectorActions[1], 2);
            alloc.Dispose();
        }
    }
}
