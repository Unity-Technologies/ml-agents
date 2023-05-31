using System;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace Unity.MLAgents.Tests
{
    public class MultiAgentGroupTests
    {
        class TestAgent : Agent
        {
            internal int _GroupId
            {
                get
                {
                    return (int)typeof(Agent).GetField("m_GroupId", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
                }
            }

            internal float _GroupReward
            {
                get
                {
                    return (float)typeof(Agent).GetField("m_GroupReward", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
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
        public void TestRegisteredAgentGroupId()
        {
            var agentGo = new GameObject("TestAgent");
            agentGo.AddComponent<TestAgent>();
            var agent = agentGo.GetComponent<TestAgent>();

            // test register
            SimpleMultiAgentGroup agentGroup1 = new SimpleMultiAgentGroup();
            agentGroup1.RegisterAgent(agent);
            Assert.AreEqual(agentGroup1.GetId(), agent._GroupId);
            Assert.IsNotNull(agent._OnAgentDisabledActions);

            // should not be able to registered to multiple groups
            SimpleMultiAgentGroup agentGroup2 = new SimpleMultiAgentGroup();
            Assert.Throws<UnityAgentsException>(
                () => agentGroup2.RegisterAgent(agent));
            Assert.AreEqual(agentGroup1.GetId(), agent._GroupId);

            // test unregister
            agentGroup1.UnregisterAgent(agent);
            Assert.AreEqual(0, agent._GroupId);
            Assert.IsNull(agent._OnAgentDisabledActions);

            // test register to another group after unregister
            agentGroup2.RegisterAgent(agent);
            Assert.AreEqual(agentGroup2.GetId(), agent._GroupId);
            Assert.IsNotNull(agent._OnAgentDisabledActions);
        }

        [Test]
        public void TestRegisterMultipleAgent()
        {
            var agentGo1 = new GameObject("TestAgent");
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();

            SimpleMultiAgentGroup agentGroup = new SimpleMultiAgentGroup();
            agentGroup.RegisterAgent(agent1); // register
            Assert.AreEqual(agentGroup.GetRegisteredAgents().Count, 1);
            agentGroup.UnregisterAgent(agent2); // unregister non-member agent
            Assert.AreEqual(agentGroup.GetRegisteredAgents().Count, 1);
            agentGroup.UnregisterAgent(agent1); // unregister
            Assert.AreEqual(agentGroup.GetRegisteredAgents().Count, 0);
            agentGroup.RegisterAgent(agent1);
            agentGroup.RegisterAgent(agent1); // duplicated register
            Assert.AreEqual(agentGroup.GetRegisteredAgents().Count, 1);
            agentGroup.RegisterAgent(agent2); // register another
            Assert.AreEqual(agentGroup.GetRegisteredAgents().Count, 2);

            // test add/set group rewards
            agentGroup.AddGroupReward(0.1f);
            Assert.AreEqual(0.1f, agent1._GroupReward);
            agentGroup.AddGroupReward(0.5f);
            Assert.AreEqual(0.6f, agent1._GroupReward);
            agentGroup.SetGroupReward(0.3f);
            Assert.AreEqual(0.3f, agent1._GroupReward);
            // unregistered agent should not receive group reward
            agentGroup.UnregisterAgent(agent1);
            agentGroup.AddGroupReward(0.2f);
            Assert.AreEqual(0.3f, agent1._GroupReward);
            Assert.AreEqual(0.5f, agent2._GroupReward);

            // dispose group should automatically unregister all
            agentGroup.Dispose();
            Assert.AreEqual(0, agent1._GroupId);
            Assert.AreEqual(0, agent2._GroupId);
        }

        [Test]
        public void TestGroupIdCounter()
        {
            SimpleMultiAgentGroup group1 = new SimpleMultiAgentGroup();
            SimpleMultiAgentGroup group2 = new SimpleMultiAgentGroup();
            // id should be unique
            Assert.AreNotEqual(group1.GetId(), group2.GetId());
        }
    }
}
