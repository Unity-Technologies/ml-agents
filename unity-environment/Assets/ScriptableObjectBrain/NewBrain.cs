using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
//    public enum PolicyType
//    {
//        Learned,
//        Human,
//        Scripted
//    }
    
    public abstract class NewBrain : ScriptableObject
    {
        [SerializeField] public BrainParameters brainParameters; // TODO :  Remove
        
        protected Dictionary<Agent, AgentInfo> agentInfo =
            new Dictionary<Agent, AgentInfo>(1024);
        
//        public PolicyType m_Policy;
        public string m_ArchetypeName;
        
        protected bool broadcast = true;

        protected MLAgents.Batcher brainBatcher;

        public abstract void InitializeBrain(Academy aca, MLAgents.Batcher batcher);
        public abstract void SendState(Agent agent, AgentInfo info);
    }
    
    

}