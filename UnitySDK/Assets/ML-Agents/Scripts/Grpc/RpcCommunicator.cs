# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
using Grpc.Core;
#endif
#if UNITY_EDITOR
using UnityEditor;
#endif
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MLAgents.CommunicatorObjects;
using System.IO;
using Google.Protobuf;

namespace MLAgents
{
    /// Responsible for communication with External using gRPC.
    public class RpcCommunicator : ICommunicator
    {
        public event QuitCommandHandler QuitCommandReceived;
        public event ResetCommandHandler ResetCommandReceived;

        /// If true, the communication is active.
        bool m_IsOpen;

        /// The default number of agents in the scene
        const int k_NumAgents = 32;

        /// Keeps track of the agents of each brain on the current step
        Dictionary<string, List<Agent>> m_CurrentAgents =
            new Dictionary<string, List<Agent>>();

        /// The current UnityRLOutput to be sent when all the brains queried the communicator
        UnityRLOutputProto m_CurrentUnityRlOutput =
            new UnityRLOutputProto();

        Dictionary<string, Dictionary<Agent, AgentAction>> m_LastActionsReceived =
            new Dictionary<string, Dictionary<Agent, AgentAction>>();

        // Brains that we have sent over the communicator with agents.
        HashSet<string> m_SentBrainKeys = new HashSet<string>();
        Dictionary<string, BrainParameters> m_UnsentBrainKeys = new Dictionary<string, BrainParameters>();


# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
        /// The Unity to External client.
        UnityToExternalProto.UnityToExternalProtoClient m_Client;
#endif
        /// The communicator parameters sent at construction
        CommunicatorInitParameters m_CommunicatorInitParameters;

        Dictionary<int, SideChannel> m_SideChannels = new Dictionary<int, SideChannel>();

        /// <summary>
        /// Initializes a new instance of the RPCCommunicator class.
        /// </summary>
        /// <param name="communicatorInitParameters">Communicator parameters.</param>
        public RpcCommunicator(CommunicatorInitParameters communicatorInitParameters)
        {
            m_CommunicatorInitParameters = communicatorInitParameters;
        }

        #region Initialization

        /// <summary>
        /// Sends the initialization parameters through the Communicator.
        /// Is used by the academy to send initialization parameters to the communicator.
        /// </summary>
        /// <returns>The External Initialization Parameters received.</returns>
        /// <param name="initParameters">The Unity Initialization Parameters to be sent.</param>
        public UnityRLInitParameters Initialize(CommunicatorInitParameters initParameters)
        {
            var academyParameters = new UnityRLInitializationOutputProto
            {
                Name = initParameters.name,
                Version = initParameters.version
            };

            UnityInputProto input;
            UnityInputProto initializationInput;
            try
            {
                initializationInput = Initialize(
                    new UnityOutputProto
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

            UpdateEnvironmentWithInput(input.RlInput);
            return initializationInput.RlInitializationInput.ToUnityRLInitParameters();
        }

        /// <summary>
        /// Adds the brain to the list of brains which will be sending information to External.
        /// </summary>
        /// <param name="brainKey">Brain key.</param>
        /// <param name="brainParameters">Brain parameters needed to send to the trainer.</param>
        public void SubscribeBrain(string brainKey, BrainParameters brainParameters)
        {
            if (m_CurrentAgents.ContainsKey(brainKey))
            {
                return;
            }
            m_CurrentAgents[brainKey] = new List<Agent>(k_NumAgents);
            m_CurrentUnityRlOutput.AgentInfos.Add(
                brainKey,
                new UnityRLOutputProto.Types.ListAgentInfoProto()
            );

            CacheBrainParameters(brainKey, brainParameters);
        }

        void UpdateEnvironmentWithInput(UnityRLInputProto rlInput)
        {
            ProcessSideChannelData(m_SideChannels, rlInput.SideChannel.ToArray());
            SendCommandEvent(rlInput.Command);

        }

        UnityInputProto Initialize(UnityOutputProto unityOutput,
            out UnityInputProto unityInput)
        {
# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
            m_IsOpen = true;
            var channel = new Channel(
                "localhost:" + m_CommunicatorInitParameters.port,
                ChannelCredentials.Insecure);

            m_Client = new UnityToExternalProto.UnityToExternalProtoClient(channel);
            var result = m_Client.Exchange(WrapMessage(unityOutput, 200));
            unityInput = m_Client.Exchange(WrapMessage(null, 200)).UnityInput;
#if UNITY_EDITOR
#if UNITY_2017_2_OR_NEWER
            EditorApplication.playModeStateChanged += HandleOnPlayModeChanged;
#else
            EditorApplication.playmodeStateChanged += HandleOnPlayModeChanged;
#endif
#endif
            return result.UnityInput;
#else
            throw new UnityAgentsException(
                "You cannot perform training on this platform.");
#endif
        }

        #endregion

        #region Destruction

        /// <summary>
        /// Close the communicator gracefully on both sides of the communication.
        /// </summary>
        public void Dispose()
        {
# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
            if (!m_IsOpen)
            {
                return;
            }

            try
            {
                m_Client.Exchange(WrapMessage(null, 400));
                m_IsOpen = false;
            }
            catch
            {
                // ignored
            }
#else
            throw new UnityAgentsException(
                "You cannot perform training on this platform.");
#endif
        }

        #endregion

        #region Sending Events

        void SendCommandEvent(CommandProto command)
        {
            switch (command)
            {
                case CommandProto.Quit:
                    {
                        QuitCommandReceived?.Invoke();
                        return;
                    }
                case CommandProto.Reset:
                    {
                        ResetCommandReceived?.Invoke();
                        return;
                    }
                default:
                    {
                        return;
                    }
            }
        }

        #endregion

        #region Sending and retreiving data

        public void DecideBatch()
        {
            if (m_CurrentAgents.Values.All(l => l.Count == 0))
            {
                return;
            }
            foreach (var brainKey in m_CurrentAgents.Keys)
            {
                using (TimerStack.Instance.Scoped("AgentInfo.ToProto"))
                {
                    if (m_CurrentAgents[brainKey].Count > 0)
                    {
                        foreach (var agent in m_CurrentAgents[brainKey])
                        {
                            // Update the sensor data on the AgentInfo
                            agent.GenerateSensorData();
                            var agentInfoProto = agent.Info.ToAgentInfoProto();
                            m_CurrentUnityRlOutput.AgentInfos[brainKey].Value.Add(agentInfoProto);
                        }

                    }
                }
            }
            SendBatchedMessageHelper();
            foreach (var brainKey in m_CurrentAgents.Keys)
            {
                m_CurrentAgents[brainKey].Clear();
            }
        }

        /// <summary>
        /// Sends the observations of one Agent. 
        /// </summary>
        /// <param name="brainKey">Batch Key.</param>
        /// <param name="agent">Agent info.</param>
        public void PutObservations(string brainKey, Agent agent)
        {
            m_CurrentAgents[brainKey].Add(agent);
        }

        /// <summary>
        /// Helper method that sends the current UnityRLOutput, receives the next UnityInput and
        /// Applies the appropriate AgentAction to the agents.
        /// </summary>
        void SendBatchedMessageHelper()
        {
            var message = new UnityOutputProto
            {
                RlOutput = m_CurrentUnityRlOutput,
            };
            var tempUnityRlInitializationOutput = GetTempUnityRlInitializationOutput();
            if (tempUnityRlInitializationOutput != null)
            {
                message.RlInitializationOutput = tempUnityRlInitializationOutput;
            }

            byte[] messageAggregated = GetSideChannelMessage(m_SideChannels);
            message.RlOutput.SideChannel = ByteString.CopyFrom(messageAggregated);

            var input = Exchange(message);
            UpdateSentBrainParameters(tempUnityRlInitializationOutput);

            foreach (var k in m_CurrentUnityRlOutput.AgentInfos.Keys)
            {
                m_CurrentUnityRlOutput.AgentInfos[k].Value.Clear();
            }

            var rlInput = input?.RlInput;

            if (rlInput?.AgentActions == null)
            {
                return;
            }

            UpdateEnvironmentWithInput(rlInput);

            m_LastActionsReceived.Clear();
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

                var agentActions = rlInput.AgentActions[brainName].ToAgentActionList();
                var numAgents = m_CurrentAgents[brainName].Count;
                var agentActionDict = new Dictionary<Agent, AgentAction>(numAgents);
                m_LastActionsReceived[brainName] = agentActionDict;
                for (var i = 0; i < numAgents; i++)
                {
                    var agent = m_CurrentAgents[brainName][i];
                    var agentAction = agentActions[i];
                    agentActionDict[agent] = agentAction;
                    agent.UpdateAgentAction(agentAction);
                }
            }
        }

        public Dictionary<Agent, AgentAction> GetActions(string key)
        {
            return m_LastActionsReceived[key];
        }

        /// <summary>
        /// Send a UnityOutput and receives a UnityInput.
        /// </summary>
        /// <returns>The next UnityInput.</returns>
        /// <param name="unityOutput">The UnityOutput to be sent.</param>
        UnityInputProto Exchange(UnityOutputProto unityOutput)
        {
# if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
            if (!m_IsOpen)
            {
                return null;
            }
            try
            {
                var message = m_Client.Exchange(WrapMessage(unityOutput, 200));
                if (message.Header.Status == 200)
                {
                    return message.UnityInput;
                }

                m_IsOpen = false;
                // Not sure if the quit command is actually sent when a
                // non 200 message is received.  Notify that we are indeed
                // quitting.
                QuitCommandReceived?.Invoke();
                return message.UnityInput;
            }
            catch
            {
                m_IsOpen = false;
                QuitCommandReceived?.Invoke();
                return null;
            }
#else
            throw new UnityAgentsException(
                "You cannot perform training on this platform.");
#endif
        }

        /// <summary>
        /// Wraps the UnityOuptut into a message with the appropriate status.
        /// </summary>
        /// <returns>The UnityMessage corresponding.</returns>
        /// <param name="content">The UnityOutput to be wrapped.</param>
        /// <param name="status">The status of the message.</param>
        static UnityMessageProto WrapMessage(UnityOutputProto content, int status)
        {
            return new UnityMessageProto
            {
                Header = new HeaderProto { Status = status },
                UnityOutput = content
            };
        }

        void CacheBrainParameters(string brainKey, BrainParameters brainParameters)
        {
            if (m_SentBrainKeys.Contains(brainKey))
            {
                return;
            }

            // TODO We should check that if m_unsentBrainKeys has brainKey, it equals brainParameters
            m_UnsentBrainKeys[brainKey] = brainParameters;
        }

        UnityRLInitializationOutputProto GetTempUnityRlInitializationOutput()
        {
            UnityRLInitializationOutputProto output = null;
            foreach (var brainKey in m_UnsentBrainKeys.Keys)
            {
                if (m_CurrentUnityRlOutput.AgentInfos.ContainsKey(brainKey))
                {
                    if (output == null)
                    {
                        output = new UnityRLInitializationOutputProto();
                    }

                    var brainParameters = m_UnsentBrainKeys[brainKey];
                    output.BrainParameters.Add(brainParameters.ToProto(brainKey, true));
                }
            }

            return output;
        }

        void UpdateSentBrainParameters(UnityRLInitializationOutputProto output)
        {
            if (output == null)
            {
                return;
            }

            foreach (var brainProto in output.BrainParameters)
            {
                m_SentBrainKeys.Add(brainProto.BrainName);
                m_UnsentBrainKeys.Remove(brainProto.BrainName);
            }
        }

        #endregion


        #region Handling side channels

        /// <summary>
        /// Registers a side channel to the communicator. The side channel will exchange 
        /// messages with its Python equivalent.
        /// </summary>
        /// <param name="sideChannel"> The side channel to be registered.</param>
        public void RegisterSideChannel(SideChannel sideChannel)
        {
            if (m_SideChannels.ContainsKey(sideChannel.ChannelType()))
            {
                throw new UnityAgentsException(string.Format(
                "A side channel with type index {} is already registered. You cannot register multiple " +
                "side channels of the same type."));
            }
            m_SideChannels.Add(sideChannel.ChannelType(), sideChannel);
        }

        /// <summary>
        /// Grabs the messages that the registered side channels will send to Python at the current step
        /// into a singe byte array.
        /// </summary>
        /// <param name="sideChannels"> A dictionary of channel type to channel.</param>
        /// <returns></returns>
        public static byte[] GetSideChannelMessage(Dictionary<int, SideChannel> sideChannels)
        {
            using (var memStream = new MemoryStream())
            {
                using (var binaryWriter = new BinaryWriter(memStream))
                {
                    foreach (var sideChannel in sideChannels.Values)
                    {
                        var messageList = sideChannel.MessageQueue;
                        foreach (var message in messageList)
                        {
                            binaryWriter.Write(sideChannel.ChannelType());
                            binaryWriter.Write(message.Count());
                            binaryWriter.Write(message);
                        }
                        sideChannel.MessageQueue.Clear();
                    }
                    return memStream.ToArray();
                }
            }
        }

        /// <summary>
        /// Separates the data received from Python into individual messages for each registered side channel.
        /// </summary>
        /// <param name="sideChannels">A dictionary of channel type to channel.</param>
        /// <param name="dataReceived">The byte array of data received from Python.</param>
        public static void ProcessSideChannelData(Dictionary<int, SideChannel> sideChannels, byte[] dataReceived)
        {
            if (dataReceived.Length == 0)
            {
                return;
            }
            using (var memStream = new MemoryStream(dataReceived))
            {
                using (var binaryReader = new BinaryReader(memStream))
                {
                    while (memStream.Position < memStream.Length)
                    {
                        int channelType = 0;
                        byte[] message = null;
                        try
                        {
                            channelType = binaryReader.ReadInt32();
                            var messageLength = binaryReader.ReadInt32();
                            message = binaryReader.ReadBytes(messageLength);
                        }
                        catch (Exception ex)
                        {
                            throw new UnityAgentsException(
                                "There was a problem reading a message in a SideChannel. Please make sure the " +
                                "version of MLAgents in Unity is compatible with the Python version. Original error : "
                                + ex.Message);
                        }
                        if (sideChannels.ContainsKey(channelType))
                        {
                            sideChannels[channelType].OnMessageReceived(message);
                        }
                        else
                        {
                            Debug.Log(string.Format(
                                "Unknown side channel data received. Channel type "
                                + ": {0}", channelType));
                        }
                    }
                }
            }
        }

        #endregion

#if UNITY_EDITOR
#if UNITY_2017_2_OR_NEWER
        /// <summary>
        /// When the editor exits, the communicator must be closed
        /// </summary>
        /// <param name="state">State.</param>
        void HandleOnPlayModeChanged(PlayModeStateChange state)
        {
            // This method is run whenever the playmode state is changed.
            if (state == PlayModeStateChange.ExitingPlayMode)
            {
                Dispose();
            }
        }

#else
        /// <summary>
        /// When the editor exits, the communicator must be closed
        /// </summary>
        private void HandleOnPlayModeChanged()
        {
            // This method is run whenever the playmode state is changed.
            if (!EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Close();
            }
        }

#endif
#endif
    }
}
