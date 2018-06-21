using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// CoreBrain which decides actions via communication with an external system such as Python.
    public class CoreBrainExternal : ScriptableObject, CoreBrain
    {
        /**< Reference to the brain that uses this CoreBrainExternal */
        public Brain brain;

        Batcher brainBatcher;

        /// Creates the reference to the brain
        public void SetBrain(Brain b)
        {
            brain = b;
        }

        /// Generates the communicator for the Academy if none was present and
        ///  subscribe to ExternalCommunicator if it was present.
        public void InitializeCoreBrain(Batcher brainBatcher)
        {
            if (brainBatcher == null)
            {
                brainBatcher = null;
                throw new UnityAgentsException($"The brain {brain.gameObject.name} was set to" + " External mode" +
                                               " but Unity was unable to read the" + " arguments passed at launch.");
            }
            else
            {
                this.brainBatcher = brainBatcher;
                this.brainBatcher.SubscribeBrain(brain.gameObject.name);
            }

        }

        /// Uses the communicator to retrieve the actions, memories and values and
        ///  sends them to the agents
        public void DecideAction(Dictionary<Agent, AgentInfo> agentInfo)
        {
            if (brainBatcher != null)
            {
                brainBatcher.SendBrainInfo(brain.gameObject.name, agentInfo);
            }
        }

        /// Nothing needs to appear in the inspector 
        public void OnInspector()
        {

        }
    }
}
