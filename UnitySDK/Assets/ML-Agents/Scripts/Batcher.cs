using System.Collections.Generic;
using System.Linq;
using System;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// The batcher is an RL specific class that makes sure that the information each object in
    /// Unity (Academy and Brains) wants to send to External is appropriately batched together
    /// and sent only when necessary.
    ///
    /// The Batcher will only send a Message to the Communicator when either :
    ///     1 - The academy is done
    ///     2 - At least one brain has data to send
    ///
    /// At each step, the batcher will keep track of the brains that queried the batcher for that
    /// step. The batcher can only send the batched data when all the Brains have queried the
    /// Batcher.
    /// </summary>
    public class Batcher
    {
        /// The default number of agents in the scene
        private const int k_NumAgents = 32;

        /// Keeps track of which brains have data to send on the current step
        Dictionary<string, bool> m_HasData =
            new Dictionary<string, bool>();

        /// Keeps track of which brains queried the batcher on the current step
        Dictionary<string, bool> m_HasQueried =
            new Dictionary<string, bool>();

        /// Keeps track of the agents of each brain on the current step
        Dictionary<string, List<Agent>> m_CurrentAgents =
            new Dictionary<string, List<Agent>>();

        /// The Communicator of the batcher, sends a message at most once per step
        ICommunicator m_Communicator;

        /// The current UnityRLOutput to be sent when all the brains queried the batcher
        CommunicatorObjects.UnityRLOutput m_CurrentUnityRlOutput =
            new CommunicatorObjects.UnityRLOutput();

        /// Keeps track of last CommandProto sent by External
        CommunicatorObjects.CommandProto m_Command;

        /// Keeps track of last EnvironmentParametersProto sent by External
        CommunicatorObjects.EnvironmentParametersProto m_EnvironmentParameters;

        /// Keeps track of last training mode sent by External
        bool m_IsTraining;

        /// Keeps track of the number of messages received
        private ulong m_MessagesReceived;

        /// <summary>
        /// Initializes a new instance of the Batcher class.
        /// </summary>
        /// <param name="communicator">The communicator to be used by the batcher.</param>
        public Batcher(ICommunicator communicator)
        {
            m_Communicator = communicator;
        }

        /// <summary>
        /// Sends the academy parameters through the Communicator.
        /// Is used by the academy to send the AcademyParameters to the communicator.
        /// </summary>
        /// <returns>The External Initialization Parameters received.</returns>
        /// <param name="academyParameters">The Unity Initialization Parameters to be sent.</param>
        public CommunicatorObjects.UnityRLInitializationInput SendAcademyParameters(
            CommunicatorObjects.UnityRLInitializationOutput academyParameters)
        {
            CommunicatorObjects.UnityInput input;
            var initializationInput = new CommunicatorObjects.UnityInput();
            try
            {
                initializationInput = m_Communicator.Initialize(
                    new CommunicatorObjects.UnityOutput
                    {
                        RlInitializationOutput = academyParameters
                    },
                    out input);
            }
            catch
            {
                var exceptionMessage = "The Communicator was unable to connect. Please make sure the External " +
                                       "process is ready to accept communication with Unity.";

                // Check for common error condition and add details to the exception message.
                var httpProxy = Environment.GetEnvironmentVariable("HTTP_PROXY");
                var httpsProxy = Environment.GetEnvironmentVariable("HTTPS_PROXY");
                if (httpProxy != null || httpsProxy != null)
                {
                    exceptionMessage += " Try removing HTTP_PROXY and HTTPS_PROXY from the" +
                                        "environment variables and try again.";
                }
                throw new UnityAgentsException(exceptionMessage);
            }

            var firstRlInput = input.RlInput;
            m_Command = firstRlInput.Command;
            m_EnvironmentParameters = firstRlInput.EnvironmentParameters;
            m_IsTraining = firstRlInput.IsTraining;
            return initializationInput.RlInitializationInput;
        }

        /// <summary>
        /// Gets the command. Is used by the academy to get reset or quit signals.
        /// </summary>
        /// <returns>The current command.</returns>
        public CommunicatorObjects.CommandProto GetCommand()
        {
            return m_Command;
        }

        /// <summary>
        /// Gets the number of messages received so far. Can be used to check for new messages.
        /// </summary>
        /// <returns>The number of messages received since start of the simulation</returns>
        public ulong GetNumberMessageReceived()
        {
            return m_MessagesReceived;
        }

        /// <summary>
        /// Gets the environment parameters. Is used by the academy to update
        /// the environment parameters.
        /// </summary>
        /// <returns>The environment parameters.</returns>
        public CommunicatorObjects.EnvironmentParametersProto GetEnvironmentParameters()
        {
            return m_EnvironmentParameters;
        }

        /// <summary>
        /// Gets the last training_mode flag External sent
        /// </summary>
        /// <returns><c>true</c>, if training mode is requested, <c>false</c> otherwise.</returns>
        public bool GetIsTraining()
        {
            return m_IsTraining;
        }

        /// <summary>
        /// Adds the brain to the list of brains which will be sending information to External.
        /// </summary>
        /// <param name="brainKey">Brain key.</param>
        public void SubscribeBrain(string brainKey)
        {
            m_HasQueried[brainKey] = false;
            m_HasData[brainKey] = false;
            m_CurrentAgents[brainKey] = new List<Agent>(k_NumAgents);
            m_CurrentUnityRlOutput.AgentInfos.Add(
                brainKey,
                new CommunicatorObjects.UnityRLOutput.Types.ListAgentInfoProto());
        }

        /// <summary>
        /// Sends the brain info. If at least one brain has an agent in need of
        /// a decision or if the academy is done, the data is sent via
        /// Communicator. Else, a new step is realized. The data can only be
        /// sent once all the brains that subscribed to the batcher have tried
        /// to send information.
        /// </summary>
        /// <param name="brainKey">Brain key.</param>
        /// <param name="agentInfo">Agent info.</param>
        public void SendBrainInfo(
            string brainKey, Dictionary<Agent, AgentInfo> agentInfo)
        {
            // If no communicator is initialized, the Batcher will not transmit
            // BrainInfo
            if (m_Communicator == null)
            {
                return;
            }

            // The brain tried called GiveBrainInfo, update m_hasQueried
            m_HasQueried[brainKey] = true;
            // Populate the currentAgents dictionary
            m_CurrentAgents[brainKey].Clear();
            foreach (var agent in agentInfo.Keys)
            {
                m_CurrentAgents[brainKey].Add(agent);
            }

            // If at least one agent has data to send, then append data to
            // the message and update hasSentState
            if (m_CurrentAgents[brainKey].Count > 0)
            {
                foreach (var agent in m_CurrentAgents[brainKey])
                {
                    var agentInfoProto = agentInfo[agent].ToProto();
                    m_CurrentUnityRlOutput.AgentInfos[brainKey].Value.Add(agentInfoProto);
                    // Avoid visual obs memory leak. This should be called AFTER we are done with the visual obs.
                    // e.g. after recording them to demo and using them for inference.
                    agentInfo[agent].ClearVisualObs();
                }

                m_HasData[brainKey] = true;
            }

            // If any agent needs to send data, then the whole message
            // must be sent
            if (m_HasQueried.Values.All(x => x))
            {
                if (m_HasData.Values.Any(x => x))
                {
                    SendBatchedMessageHelper();
                }

                // The message was just sent so we must reset hasSentState and
                // triedSendState
                foreach (var k in m_CurrentAgents.Keys)
                {
                    m_HasData[k] = false;
                    m_HasQueried[k] = false;
                }
            }
        }

        /// <summary>
        /// Helper method that sends the current UnityRLOutput, receives the next UnityInput and
        /// Applies the appropriate AgentAction to the agents.
        /// </summary>
        void SendBatchedMessageHelper()
        {
            var input = m_Communicator.Exchange(
                new CommunicatorObjects.UnityOutput
                {
                    RlOutput = m_CurrentUnityRlOutput
                });
            m_MessagesReceived += 1;

            foreach (var k in m_CurrentUnityRlOutput.AgentInfos.Keys)
            {
                m_CurrentUnityRlOutput.AgentInfos[k].Value.Clear();
            }

            if (input == null)
            {
                m_Command = CommunicatorObjects.CommandProto.Quit;
                return;
            }

            var rlInput = input.RlInput;

            if (rlInput == null)
            {
                m_Command = CommunicatorObjects.CommandProto.Quit;
                return;
            }

            m_Command = rlInput.Command;
            m_EnvironmentParameters = rlInput.EnvironmentParameters;
            m_IsTraining = rlInput.IsTraining;

            if (rlInput.AgentActions == null)
            {
                return;
            }

            foreach (var brainName in rlInput.AgentActions.Keys)
            {
                if (!m_CurrentAgents[brainName].Any())
                {
                    continue;
                }

                if (!rlInput.AgentActions[brainName].Value.Any())
                {
                    continue;
                }

                for (var i = 0; i < m_CurrentAgents[brainName].Count; i++)
                {
                    var agent = m_CurrentAgents[brainName][i];
                    var action = rlInput.AgentActions[brainName].Value[i];
                    agent.UpdateVectorAction(action.VectorActions.ToArray());
                    agent.UpdateMemoriesAction(action.Memories.ToList());
                    agent.UpdateTextAction(action.TextActions);
                    agent.UpdateValueAction(action.Value);
                    agent.UpdateCustomAction(action.CustomAction);
                }
            }
        }
    }
}
