using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace MLAgents
{


    public class ScriptableBrain : Brain
    {
        [SerializeField]
        [HideInInspector]
        public Decision decision;
#if UNITY_EDITOR
        [HideInInspector]
        public MonoScript decisionScript;
#endif
        [SerializeField]
        [HideInInspector]
        public string c_decision;

        
        public void OnValidate()
        {
#if UNITY_EDITOR
            if (decisionScript != null)
            {
                c_decision = decisionScript.GetClass().Name;
            }
            else
            {
                c_decision = "";
            }

//            if (c_decision != c_oldDecision)
//            {
//                Debug.Log(c_decision);
//                decision = CreateInstance(c_decision) as Decision;
//                decision.brainParameters = brainParameters;
//                Debug.Log(decision);
//                c_oldDecision = c_decision;
//            }
#endif
        }

        protected override void DecideAction()
        {

            if ((c_decision != null) && decision == null)
            {
                decision = CreateInstance(c_decision) as Decision;
                decision.brainParameters = brainParameters;
            }
            
            if (brainBatcher != null)
            {
                brainBatcher.SendBrainInfo(this.name, agentInfo);
            }
            
            if (isExternal)
            {
                agentInfo.Clear();
                return;
            }

            foreach (Agent agent in agentInfo.Keys)
            {
                agent.UpdateVectorAction(decision.Decide(
                    agentInfo[agent].stackedVectorObservation,
                    agentInfo[agent].visualObservations,
                    agentInfo[agent].reward,
                    agentInfo[agent].done,
                    agentInfo[agent].memories));
            }

            foreach (Agent agent in agentInfo.Keys)
            {
                agent.UpdateMemoriesAction(decision.MakeMemory(
                    agentInfo[agent].stackedVectorObservation,
                    agentInfo[agent].visualObservations,
                    agentInfo[agent].reward,
                    agentInfo[agent].done,
                    agentInfo[agent].memories));
            }
            agentInfo.Clear();
        }
    }

}
