using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MLAgents
{

    public class BrainBatcher
    {
        const int NUM_AGENTS = 32;
        Dictionary<string, bool> hasSentState = new Dictionary<string, bool>();
        Dictionary<string, bool> triedSendState = new Dictionary<string, bool>();

        Dictionary<string, List<Agent>> currentAgents = new Dictionary<string, List<Agent>>();
        Communicator.Communicator communicator;
        Communicator.UnityOutput unityOutput = new Communicator.UnityOutput();

        // TO delete
        Academy academy;

        public BrainBatcher(Communicator.Communicator communicator)
        {
            this.communicator = communicator;
            // This needs to disapear, the done flag of the academy should be accessible by everything
            academy = Object.FindObjectOfType<Academy>() as Academy;

        }

        /// <summary>
        /// Adds the brain to the list of brains which have already decided their
        /// actions.
        /// </summary>
        /// <param name="brainKey">Brain key.</param>
        public void SubscribeBrain(string brainKey)
        {
            triedSendState[brainKey] = false;
            hasSentState[brainKey] = false;
            currentAgents[brainKey] = new List<Agent>(NUM_AGENTS);
            unityOutput.AgentInfos.Add(brainKey, new Communicator.UnityOutput.Types.ListAgentInfo());
        }

        /// <summary>
        /// Gives the brain info.
        /// </summary>
        /// <param name="brainKey">Brain key.</param>
        /// <param name="agentInfo">Agent info.</param>
        public void GiveBrainInfo(string brainKey, Dictionary<Agent, AgentInfo> agentInfo)
        {
            //TODO : Find a way to remove this academy




            if (communicator == null)
            {
                return;
            }

            triedSendState[brainKey] = true;
            currentAgents[brainKey].Clear();
            foreach (Agent agent in agentInfo.Keys)
            {
                currentAgents[brainKey].Add(agent);
            }
            if (currentAgents[brainKey].Count > 0)
            {
                Communicator.UnityOutput.Types.ListAgentInfo listAgentInfo =
                    new Communicator.UnityOutput.Types.ListAgentInfo();
                foreach (Agent agent in currentAgents[brainKey])
                {
                    Communicator.AgentInfo ai = new Communicator.AgentInfo();
                    ai.VectorObservation.AddRange(agentInfo[agent].vectorObservation);
                    ai.StackedVectorObservation.AddRange(agentInfo[agent].stackedVectorObservation);
                    ai.StoredVectorActions.AddRange(agentInfo[agent].storedVectorActions);
                    //TODO : Visual Observations and memories and text action
                    ai.Reward = agentInfo[agent].reward;
                    ai.MaxStepReached = agentInfo[agent].maxStepReached;
                    ai.Done = agentInfo[agent].done;
                    ai.Id = agentInfo[agent].id;
                    listAgentInfo.Value.Add(ai);
                }

                //TODO :: If the key is present, it will raise an error
                unityOutput.AgentInfos[brainKey] = listAgentInfo;
                hasSentState[brainKey] = true;

                if (triedSendState.Values.All(x => x))
                {
                    if (hasSentState.Values.Any(x => x) || academy.IsDone())
                    {
                        //Debug.Log("Received the new input");
                        var input = communicator.SendOuput(unityOutput);

                        // TODO : Send the actions of the input to the agents
                        if (input.AgentActions != null)
                        {
                            //Debug.Log(input.AgentActions["Ball3DBrain"].Value.Count);
                            foreach (string k in input.AgentActions.Keys)
                            {
                                if (currentAgents[k].Count() == 0)
                                {
                                    continue;
                                }
                                if (input.AgentActions[k].Value.Count == 0)
                                {
                                    continue;
                                }
                                for (int i = 0; i < currentAgents[k].Count(); i++)
                                {
                                    currentAgents[k][i].UpdateVectorAction(input.AgentActions[k].Value[i].VectorActions.ToArray());
                                }
                            }
                        }

                        foreach (string k in currentAgents.Keys)
                        {
                            hasSentState[k] = false;
                            triedSendState[k] = false;
                        }
                    }
                }

            }
        }
    }
}



///// Listens for actions, memories, and values and sends them 
///// to the corrensponding brains.
//public void UpdateActions()
//{
//// TODO

//foreach (Brain brain in brains)
//{
//    if (brain.brainType == BrainType.External)
//    {
//        var brainName = brain.gameObject.name;

//        if (current_agents[brainName].Count() == 0)
//        {
//            continue;
//        }


//        for (int i = 0; i < current_agents[brainName].Count(); i++)
//        {
//            current_agents[brainName][i].UpdateVectorAction(comm.inputs.AgentActions[brainName].Value[i].VectorActions.ToArray());


//        }

//    }
//}
