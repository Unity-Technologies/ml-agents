using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{


    public class ScriptableBrain : NewBrain
    {

        /**< Reference to the Decision component used to decide the actions */
//        public Decision decision = new RandomDecision();

        public override void InitializeBrain(Academy aca, MLAgents.Batcher batcher)
        {
            aca.BrainDecideAction += DecideAction;
            if ((brainBatcher == null)
                || (!broadcast))
            {
                this.brainBatcher = null;
            }
            else
            {
                this.brainBatcher = batcher;
                this.brainBatcher.SubscribeBrain(this.name);
            }
        }

        void DecideAction()
        {
            if (brainBatcher != null)
            {
                brainBatcher.SendBrainInfo(this.name, agentInfo);
            }

            foreach (Agent agent in agentInfo.Keys)
            {
                Decision decision = agent.gameObject.GetComponent<Decision>();
                agent.UpdateVectorAction(decision.Decide(
                    agentInfo[agent].stackedVectorObservation,
                    agentInfo[agent].visualObservations,
                    agentInfo[agent].reward,
                    agentInfo[agent].done,
                    agentInfo[agent].memories));
            }

            foreach (Agent agent in agentInfo.Keys)
            {
                Decision decision = agent.gameObject.GetComponent<Decision>();
                agent.UpdateMemoriesAction(decision.MakeMemory(
                    agentInfo[agent].stackedVectorObservation,
                    agentInfo[agent].visualObservations,
                    agentInfo[agent].reward,
                    agentInfo[agent].done,
                    agentInfo[agent].memories));
            }
            agentInfo.Clear();
        }
        
        public override void SendState(Agent agent, AgentInfo info)
        {
            agentInfo.Add(agent, info);

        }
    }

}
