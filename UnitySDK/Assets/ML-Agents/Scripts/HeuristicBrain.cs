using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace MLAgents
{
    /// Brain which decides actions using developer-provided Decision script.
    [CreateAssetMenu(fileName = "NewHeuristicBrain", menuName = "ML-Agents/Heuristic Brain")]
    public class HeuristicBrain : Brain
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
#endif
        }
        
        /// Loads the tensorflow graph model to generate a TFGraph object
        protected override void Initialize()
        {
            if ((c_decision != null) && decision == null)
            {
                decision = CreateInstance(c_decision) as Decision;
                decision.brainParameters = brainParameters;
            }
        }


        /// Uses the Decision Component to decide that action to take
        protected override void DecideAction()
        {
            base.DecideAction();
            
            if (decision == null)
            {
                throw new UnityAgentsException(
                    "The Brain is set to Heuristic, but no decision script attached to it");
            }

            foreach (Agent agent in agentInfos.Keys)
            {
                agent.UpdateVectorAction(decision.Decide(
                    agentInfos[agent].stackedVectorObservation,
                    agentInfos[agent].visualObservations,
                    agentInfos[agent].reward,
                    agentInfos[agent].done,
                    agentInfos[agent].memories));

            }

            foreach (Agent agent in agentInfos.Keys)
            {
                agent.UpdateMemoriesAction(decision.MakeMemory(
                    agentInfos[agent].stackedVectorObservation,
                    agentInfos[agent].visualObservations,
                    agentInfos[agent].reward,
                    agentInfos[agent].done,
                    agentInfos[agent].memories));
            }

            agentInfos.Clear();
        }
    }
}
