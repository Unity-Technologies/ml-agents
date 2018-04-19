using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Google.Protobuf;

namespace MLAgents
{

    public class Batcher
    {
        const int NUM_AGENTS = 32;
        Dictionary<string, bool> hasSentState =
            new Dictionary<string, bool>();
        Dictionary<string, bool> triedSendState =
            new Dictionary<string, bool>();

        Dictionary<string, List<Agent>> currentAgents =
            new Dictionary<string, List<Agent>>();
        Communicator.Communicator communicator;
        Communicator.UnityRLOutput unityOutput =
            new Communicator.UnityRLOutput();

        bool academyDone;
        Communicator.Command command;
        Communicator.EnvironmentParameters environmentParameters;
        bool isTraining;

        public Batcher(Communicator.Communicator communicator)
        {
            this.communicator = communicator;
        }

        /// <summary>
        /// Gives the academy parameters. Is used by the academy to send the
        /// AcademyParameters to the communicator.
        /// </summary>
        /// <param name="academyParameters">Academy parameters.</param>
        public void GiveAcademyParameters(
            Communicator.AcademyParameters academyParameters)
        {
            Communicator.UnityRLInput input;
            communicator.Initialize(academyParameters, out input);
            command = input.Command;
            environmentParameters = input.EnvironmentParameters;
            isTraining = input.IsTraining;
        }

        /// <summary>
        /// Adds the done flag of the academy to the next output to be sent
        /// to the communicator.
        /// </summary>
        /// <param name="done">If set to <c>true</c> 
        /// The academy is done.</param>
        public void GiveAcademyDone(bool done)
        {
            academyDone = done;
        }

        /// <summary>
        /// Gets the command. Is used by the academy to get reset or quit
        /// signal.
        /// </summary>
        /// <returns>The command.</returns>
        public Communicator.Command GetCommand()
        {
            return command;
        }

        /// <summary>
        /// Gets the environment parameters. Is used by the academy to update
        /// the environment parameters.
        /// </summary>
        /// <returns>The environment parameters.</returns>
        public Communicator.EnvironmentParameters GetEnvironmentParameters()
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
            unityOutput.AgentInfos.Add(
                brainKey,
                new Communicator.UnityRLOutput.Types.ListAgentInfo());
        }

        /// <summary>
        /// Converts a AgentInfo to a protobuffer generated AgentInfo
        /// </summary>
        /// <returns>The Proto agentInfo.</returns>
        /// <param name="info">The AgentInfo to convert.</param>
        public static Communicator.AgentInfo AgentInfoConvertor(AgentInfo info)
        {

            Communicator.AgentInfo ai = new Communicator.AgentInfo
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
                ai.VisualObservations.Add(
                    ByteString.CopyFrom(obs.EncodeToJPG())
                );
            }
            return ai;
        }

        /// <summary>
        /// Converts a Brain into to a Protobuff BrainInfo so it can be sent
        /// </summary>
        /// <returns>The parameters convertor.</returns>
        /// <param name="bp">The BrainParameters.</param>
        /// <param name="name">The name of the brain.</param>
        /// <param name="type">The type of brain.</param>
        public static Communicator.BrainParameters BrainParametersConvertor(
            BrainParameters bp, string name, Communicator.BrainType type)
        {

            Communicator.BrainParameters brainParameters =
                new Communicator.BrainParameters
                {
                    VectorObservationSize = bp.vectorObservationSize,
                    NumStackedVectorObservations = bp.numStackedVectorObservations,
                    VectorActionSize = bp.vectorActionSize,
                    VectorActionSpaceType =
                    (Communicator.SpaceType)bp.vectorActionSpaceType,
                    VectorObservationSpaceType =
                    (Communicator.SpaceType)bp.vectorObservationSpaceType,
                    BrainName = name,
                    BrainType = type
                };
            brainParameters.VectorActionDescriptions.AddRange(
                bp.vectorActionDescriptions);
            foreach (resolution res in bp.cameraResolutions)
            {
                brainParameters.CameraResolutions.Add(
                    new Communicator.Resolution
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
        public void GiveBrainInfo(
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
                    Communicator.AgentInfo ai =
                        AgentInfoConvertor(agentInfo[agent]);
                    unityOutput.AgentInfos[brainKey].Value.Add(ai);
                }
                hasSentState[brainKey] = true;
            }

            // If any agent needs to send data, then the whole message
            // must be sent
            if (triedSendState.Values.All(x => x))
            {
                if (hasSentState.Values.Any(x => x) || academyDone)
                {
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
            var input = communicator.SendOuput(unityOutput);

            foreach (string k in unityOutput.AgentInfos.Keys)
            {
                unityOutput.AgentInfos[k].Value.Clear();
            }
            if (input == null)
            {
                command = Communicator.Command.Quit;
                return;
            }

            command = input.Command;
            environmentParameters = input.EnvironmentParameters;
            isTraining = input.IsTraining;

            if (input.AgentActions != null)
            {
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
                        var agent = currentAgents[k][i];
                        var action = input.AgentActions[k].Value[i];
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



