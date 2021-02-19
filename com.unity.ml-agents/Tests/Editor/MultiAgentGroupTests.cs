// using System;
// using System.Linq;
// using System.Collections.Generic;
using Unity.MLAgents;
using System;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using Unity;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents.Tests
{
    public class MultiAgentGroupTests
    {
        public class TestingMultiAgentGroup : IMultiAgentGroup
        {
            readonly int m_Id = MultiAgentGroupIdCounter.GetGroupId();

            /// <inheritdoc />
            public void RegisterAgent(Agent agent)
            {
                agent.SetMultiAgentGroup(this);
                agent.OnAgentDisabled += UnregisterAgent;
            }

            /// <inheritdoc />
            public void UnregisterAgent(Agent agent)
            {
                agent.SetMultiAgentGroup(null);
                agent.OnAgentDisabled -= UnregisterAgent;
            }
            public int GetId()
            {
                return m_Id;
            }
        }

        class TestAgent : Agent
        {
            internal int _GroupId
            {
                get
                {
                    return (int)typeof(Agent).GetField("m_GroupId", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
                }
            }

            internal Action<Agent> _OnAgentDisabledActions
            {
                get
                {
                    return (Action<Agent>)typeof(Agent).GetField("OnAgentDisabled", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
                }
            }
        }

        [Test]
        public void TestRegisterAgent()
        {
            TestingMultiAgentGroup agentGroup = new TestingMultiAgentGroup();
            var agentGo = new GameObject("TestAgent");
            agentGo.AddComponent<TestAgent>();
            var agent = agentGo.GetComponent<TestAgent>();

            // test register
            agentGroup.RegisterAgent(agent);
            Assert.AreEqual(agentGroup.GetId(), agent._GroupId);
            Assert.IsNotNull(agent._OnAgentDisabledActions);

            // should not be able to registered to multiple groups
            TestingMultiAgentGroup agentGroup2 = new TestingMultiAgentGroup();
            Assert.Throws<UnityAgentsException>(
                () => agentGroup2.RegisterAgent(agent));
            Assert.AreEqual(agentGroup.GetId(), agent._GroupId);

            // test unregister
            agentGroup.UnregisterAgent(agent);
            Assert.AreEqual(0, agent._GroupId);
            Assert.IsNull(agent._OnAgentDisabledActions);

            // test register to another group
            agentGroup2.RegisterAgent(agent);
            Assert.AreEqual(agentGroup2.GetId(), agent._GroupId);
            Assert.IsNotNull(agent._OnAgentDisabledActions);
        }

        [Test]
        public void TestGroupIdCounter()
        {
            TestingMultiAgentGroup group1 = new TestingMultiAgentGroup();
            TestingMultiAgentGroup group2 = new TestingMultiAgentGroup();
            // id should be unique
            Assert.AreNotEqual(group1.GetId(), group2.GetId());
        }
    }
}
