using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Google.Protobuf;

namespace MLAgents
{

    public class Batcher
    {
        private const int NUM_AGENTS = 32;
        Dictionary<string, bool> hasSentState =
            new Dictionary<string, bool>();
        Dictionary<string, bool> triedSendState =
            new Dictionary<string, bool>();

        Dictionary<string, List<Agent>> currentAgents =
            new Dictionary<string, List<Agent>>();
        Communicator communicator;
        CommunicatorObjects.UnityRLOutput currentUnityRLOutput =
                               new CommunicatorObjects.UnityRLOutput();

        bool academyDone;
        CommunicatorObjects.CommandProto command;
        CommunicatorObjects.EnvironmentParametersProto environmentParameters;
        bool isTraining;

        public Batcher(Communicator communicator)
        {
            this.communicator = communicator;
        }

        /// <summary>
        /// Gives the academy parameters. Is used by the academy to send the
        /// AcademyParameters to the communicator.
        /// </summary>
        /// <returns>The python parameters.</returns>
        /// <param name="academyParameters">Academy parameters.</param>
        public CommunicatorObjects.UnityRLInitializationInput SendAcademyParameters(
            CommunicatorObjects.UnityRLInitializationOutput academyParameters)
        {
            CommunicatorObjects.UnityInput input;
            CommunicatorObjects.UnityInput initializationInput =
                communicator.Initialize(
                    new CommunicatorObjects.UnityOutput
                    {
                        RlInitializationOutput = academyParameters
                    },
                    out input);
            CommunicatorObjects.UnityRLInput firstRLInput = input.RlInput;
            command = firstRLInput.Command;
            environmentParameters = firstRLInput.EnvironmentParameters;
            isTraining = firstRLInput.IsTraining;
            return initializationInput.RlInitializationInput;
        }

        /// <summary>
        /// Adds the done flag of the academy to the next output to be sent
        /// to the communicator.
        /// </summary>
        /// <param name="done">If set to <c>true</c> 
        /// The academy is done.</param>
        public void RegisterAcademyDoneFlag(bool done)
        {
            academyDone = done;
        }

        /// <summary>
        /// Gets the command. Is used by the academy to get reset or quit
        /// signal.
        /// </summary>
        /// <returns>The command.</returns>
        public CommunicatorObjects.CommandProto GetCommand()
        {
            return command;
        }

        /// <summary>
        /// Gets the environment parameters. Is used by the academy to update
        /// the environment parameters.
        /// </summary>
        /// <returns>The environment parameters.</returns>
        public CommunicatorObjects.EnvironmentParametersProto GetEnvironmentParameters()
        {
            return environmentParameters;
        }

        public bool GetIsTraining()
        {
            return isTraining;
        }

        /// <summary>
        /// Adds the brain to the list of brains which have already decided
        /// their actions.
        /// </summary>
        /// <param name="brainKey">Brain key.</param>
        public void SubscribeBrain(string brainKey)
        {
            triedSendState[brainKey] = false;
            hasSentState[brainKey] = false;
            currentAgents[brainKey] = new List<Agent>(NUM_AGENTS);
            currentUnityRLOutput.AgentInfos.Add(
                brainKey,
                new CommunicatorObjects.UnityRLOutput.Types.ListAgentInfoProto());
        }

        /// <summary>
        /// Converts a AgentInfo to a protobuffer generated AgentInfo
        /// </summary>
        /// <returns>The Proto agentInfo.</returns>
        /// <param name="info">The AgentInfo to convert.</param>
        public static CommunicatorObjects.AgentInfoProto 
                                         AgentInfoConvertor(AgentInfo info)
        {

            CommunicatorObjects.AgentInfoProto agentInfoProto = 
                new CommunicatorObjects.AgentInfoProto
            {
                StackedVectorObservation = { info.stackedVectorObservation },
                StoredVectorActions = { info.storedVectorActions },
                Memories = { info.memories },
                StoredTextActions = info.storedTextActions,
                TextObservation = info.textObservation,
                Reward = info.reward,
                MaxStepReached = info.maxStepReached,
                Done = info.done,
                Id = info.id,
            };
            foreach (Texture2D obs in info.visualObservations)
            {
                agentInfoProto.VisualObservations.Add(
                    ByteString.CopyFrom(obs.EncodeToJPG())
                );
            }
            return agentInfoProto;
        }

        /// <summary>
        /// Converts a Brain into to a Protobuff BrainInfo so it can be sent
        /// </summary>
        /// <returns>The parameters convertor.</returns>
        /// <param name="bp">The BrainParameters.</param>
        /// <param name="name">The name of the brain.</param>
        /// <param name="type">The type of brain.</param>
        public static CommunicatorObjects.BrainParametersProto BrainParametersConvertor(
            BrainParameters bp, string name, CommunicatorObjects.BrainTypeProto type)
        {

            CommunicatorObjects.BrainParametersProto brainParameters =
                                   new CommunicatorObjects.BrainParametersProto
                {
                    VectorObservationSize = bp.vectorObservationSize,
                    NumStackedVectorObservations = bp.numStackedVectorObservations,
                    VectorActionSize = bp.vectorActionSize,
                    VectorActionSpaceType =
                    (CommunicatorObjects.SpaceTypeProto)bp.vectorActionSpaceType,
                    VectorObservationSpaceType =
                    (CommunicatorObjects.SpaceTypeProto)bp.vectorObservationSpaceType,
                    BrainName = name,
                    BrainType = type
                };
            brainParameters.VectorActionDescriptions.AddRange(
                bp.vectorActionDescriptions);
            foreach (resolution res in bp.cameraResolutions)
            {
                brainParameters.CameraResolutions.Add(
                    new CommunicatorObjects.ResolutionProto
                    {
                        Width = res.width,
                        Height = res.height,
                        GrayScale = res.blackAndWhite
                    });
            }
            return brainParameters;
        }

        /// <summary>
        /// Gives the brain info. If at least one brain has an agent in need of
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
            if (communicator == null)
            {
                return;
            }

            // The brain tried called GiveBrainInfo, update triedSendState
            triedSendState[brainKey] = true;
            // Populate the currentAgents dictionary
            currentAgents[brainKey].Clear();
            foreach (Agent agent in agentInfo.Keys)
            {
                currentAgents[brainKey].Add(agent);
            }
            // If at least one agent has data to send, then append data to
            // the message and update hasSentState
            if (currentAgents[brainKey].Count > 0)
            {
                foreach (Agent agent in currentAgents[brainKey])
                {
                    CommunicatorObjects.AgentInfoProto agentInfoProto =
                        AgentInfoConvertor(agentInfo[agent]);
                    currentUnityRLOutput.AgentInfos[brainKey].Value.Add(agentInfoProto);
                }
                hasSentState[brainKey] = true;
            }

            // If any agent needs to send data, then the whole message
            // must be sent
            if (triedSendState.Values.All(x => x))
            {
                if (hasSentState.Values.Any(x => x) || academyDone)
                {
                    currentUnityRLOutput.GlobalDone = academyDone;
                    SendBatchedMessageHelper();
                }
                // The message was just sent so we must reset hasSentState and
                // triedSendState
                foreach (string k in currentAgents.Keys)
                {
                    hasSentState[k] = false;
                    triedSendState[k] = false;
                }
            }
        }

        void SendBatchedMessageHelper()
        {
            var input = communicator.Exchange(
                new CommunicatorObjects.UnityOutput{
                RlOutput = currentUnityRLOutput
            });

            foreach (string k in currentUnityRLOutput.AgentInfos.Keys)
            {
                currentUnityRLOutput.AgentInfos[k].Value.Clear();
            }
            if (input == null)
            {
                command = CommunicatorObjects.CommandProto.Quit;
                return;
            }

            CommunicatorObjects.UnityRLInput rlInput = input.RlInput;

            if (rlInput == null)
            {
                command = CommunicatorObjects.CommandProto.Quit;
                return;
            }

            command = rlInput.Command;
            environmentParameters = rlInput.EnvironmentParameters;
            isTraining = rlInput.IsTraining;

            if (rlInput.AgentActions != null)
            {
                foreach (string brainName in rlInput.AgentActions.Keys)
                {
                    if (currentAgents[brainName].Count() == 0)
                    {
                        continue;
                    }
                    if (rlInput.AgentActions[brainName].Value.Count == 0)
                    {
                        continue;
                    }
                    for (int i = 0; i < currentAgents[brainName].Count(); i++)
                    {
                        var agent = currentAgents[brainName][i];
                        var action = rlInput.AgentActions[brainName].Value[i];
                        agent.UpdateVectorAction(
                            action.VectorActions.ToArray());
                        agent.UpdateMemoriesAction(
                            action.Memories.ToList());
                        agent.UpdateTextAction(
                            action.TextActions);
                    }
                }
            }
        }

    }
}



