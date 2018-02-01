using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Newtonsoft.Json;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.IO;


/// Responsible for communication with Python API.
public class ExternalCommunicator : Communicator
{

    ExternalCommand command = ExternalCommand.QUIT;
    Academy academy;

    Dictionary<string, List<Agent>> current_agents;

    List<Brain> brains;

    Dictionary<string, bool> hasSentState;
    Dictionary<string, bool> triedSendState;

    Dictionary<string, Dictionary<Agent, float[]>> storedActions;
    Dictionary<string, Dictionary<Agent, float[]>> storedMemories;
    Dictionary<string, Dictionary<Agent, float>> storedValues;

    const int messageLength = 12000;
    const int defaultNumAgents = 32;
    const int defaultNumObservations = 32;


    // For Messages
    List<int> concatenatedAgentId = new List<int>(defaultNumAgents);
    List<float> concatenatedStates = new List<float>(defaultNumAgents*defaultNumObservations);
    List<float> concatenatedRewards = new List<float>(defaultNumAgents);
    List<float> concatenatedMemories = new List<float>(defaultNumAgents * defaultNumObservations);
    List<bool> concatenatedDones = new List<bool>(defaultNumAgents);
    List<bool> concatenatedMaxes = new List<bool>(defaultNumAgents);
    List<float> concatenatedActions = new List<float>(defaultNumAgents * defaultNumObservations);

    int comPort;
    int randomSeed;
    Socket sender;
    byte[] messageHolder;
    byte[] lengthHolder;

    StreamWriter logWriter;
    string logPath;

    const string api = "API-2";

    /// Placeholder for state information to send.
    [System.Serializable]
    public struct StepMessage
    {
        public string brain_name;
        public List<int> agents;
        public List<float> states;
        public List<float> rewards;
        public List<float> actions;
        public List<float> memories;
        public List<bool> dones;
        public List<bool> maxes;
    }

    StepMessage sMessage;
    string sMessageString;

    string rMessage;

    /// Placeholder for returned message.
    struct AgentMessage
    {
        public Dictionary<string, List<float>> action { get; set; }
        public Dictionary<string, List<float>> memory { get; set; }
        public Dictionary<string, List<float>> value { get; set; }
    }

    /// Placeholder for reset parameter message
    struct ResetParametersMessage
    {
        public Dictionary<string, float> parameters { get; set; }
        public bool train_model { get; set; }
    }

    /// Consrtuctor for the External Communicator
    public ExternalCommunicator(Academy aca)
    {
        academy = aca;
        brains = new List<Brain>();
        current_agents = new Dictionary<string, List<Agent>>();

        hasSentState = new Dictionary<string, bool>();
        triedSendState = new Dictionary<string, bool>();

        storedActions = new Dictionary<string, Dictionary<Agent, float[]>>();
        storedMemories = new Dictionary<string, Dictionary<Agent, float[]>>();
        storedValues = new Dictionary<string, Dictionary<Agent, float>>();
    }

    /// Adds the brain to the list of brains which have already decided their
    /// actions.
    public void SubscribeBrain(Brain brain)
    {
        brains.Add(brain);
        hasSentState[brain.gameObject.name] = false;
    }

    /// Attempts to make handshake with external API. 
    public bool CommunicatorHandShake()
    {
        try
        {
            ReadArgs();
        }
        catch
        {
            return false;
        }
        return true;
    }

    /// Contains the logic for the initializtation of the socket.
    public void InitializeCommunicator()
    {
        Application.logMessageReceived += HandleLog;
        logPath = Path.GetFullPath(".") + "/unity-environment.log";
        logWriter = new StreamWriter(logPath, false);
        logWriter.WriteLine(System.DateTime.Now.ToString());
        logWriter.WriteLine(" ");
        logWriter.Close();
        messageHolder = new byte[messageLength];
        lengthHolder = new byte[4];

        // Create a TCP/IP  socket.  
        sender = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        sender.Connect("localhost", comPort);

        var accParamerters = new AcademyParameters();

        accParamerters.brainParameters = new List<BrainParameters>();
        accParamerters.brainNames = new List<string>();
        accParamerters.externalBrainNames = new List<string>();
        accParamerters.apiNumber = api;
        accParamerters.logPath = logPath;
        foreach (Brain b in brains)
        {
            accParamerters.brainParameters.Add(b.brainParameters);
            accParamerters.brainNames.Add(b.gameObject.name);
            if (b.brainType == BrainType.External)
            {
                accParamerters.externalBrainNames.Add(b.gameObject.name);
            }
        }
        accParamerters.AcademyName = academy.gameObject.name;
        accParamerters.resetParameters = academy.resetParameters;

        SendParameters(accParamerters);

        sMessage = new StepMessage();
    }

    void HandleLog(string logString, string stackTrace, LogType type)
    {
        logWriter = new StreamWriter(logPath, true);
        logWriter.WriteLine(type.ToString());
        logWriter.WriteLine(logString);
        logWriter.WriteLine(stackTrace);
        logWriter.Close();
    }

    /// Listens to the socket for a command and returns the corresponding
    ///  External Command.
    public void UpdateCommand()
    {
        int location = sender.Receive(messageHolder);
        string message = Encoding.ASCII.GetString(messageHolder, 0, location);
        switch (message)
        {
            case "STEP":
                command = ExternalCommand.STEP;
                break;
            case "RESET":
                command = ExternalCommand.RESET;
                break;
            case "QUIT":
                command = ExternalCommand.QUIT;
                break;
            default:
                command = ExternalCommand.QUIT;
                break;
        }
    }

    public ExternalCommand GetCommand()
    {
        return command;
    }

    public void SetCommand(ExternalCommand c)
    {
        command = c;
    }

    /// Listens to the socket for the new resetParameters
    public Dictionary<string, float> GetResetParameters()
    {
        sender.Send(Encoding.ASCII.GetBytes("CONFIG_REQUEST"));
        Receive();
        var resetParams = JsonConvert.DeserializeObject<ResetParametersMessage>(rMessage);
        academy.isInference = !resetParams.train_model;
        return resetParams.parameters;
    }


    /// Used to read Python-provided environment parameters
    private void ReadArgs()
    {
        string[] args = System.Environment.GetCommandLineArgs();
        var inputPort = "";
        var inputSeed = "";
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--port")
            {
                inputPort = args[i + 1];
            }
            if (args[i] == "--seed")
            {
                inputSeed = args[i + 1];
            }
        }

        comPort = int.Parse(inputPort);
        randomSeed = int.Parse(inputSeed);
        Random.InitState(randomSeed);
    }

    /// Sends Academy parameters to external agent
    private void SendParameters(AcademyParameters envParams)
    {
        string envMessage = JsonConvert.SerializeObject(envParams, Formatting.Indented);
        sender.Send(Encoding.ASCII.GetBytes(envMessage));
    }

    /// Receives messages from external agent
    private void Receive()
    {
        int location = sender.Receive(messageHolder);
        rMessage = Encoding.ASCII.GetString(messageHolder, 0, location);
    }

    /// Receives a message and can reconstruct a message if was too long
    private string ReceiveAll()
    {
        sender.Receive(lengthHolder);
        int totalLength = System.BitConverter.ToInt32(lengthHolder, 0);
        int location = 0;
        rMessage = "";
        while (location != totalLength)
        {
            int fragment = sender.Receive(messageHolder);
            location += fragment;
            rMessage += Encoding.ASCII.GetString(messageHolder, 0, fragment);
        }
        return rMessage;
    }

    /// Ends connection and closes environment
    private void OnApplicationQuit()
    {
        sender.Close();
        sender.Shutdown(SocketShutdown.Both);
    }

    /// Contains logic for coverting texture into bytearray to send to 
    /// external agent.
    private byte[] TexToByteArray(Texture2D tex)
    {
        byte[] bytes = tex.EncodeToPNG();
        Object.DestroyImmediate(tex);
        Resources.UnloadUnusedAssets();
        return bytes;
    }

    private byte[] AppendLength(byte[] input)
    {
        byte[] newArray = new byte[input.Length + 4];
        input.CopyTo(newArray, 4);
        System.BitConverter.GetBytes(input.Length).CopyTo(newArray, 0);
        return newArray;
    }

    /// Collects the information from the brains and sends it accross the socket
    public void GiveBrainInfo(Brain brain, Dictionary<Agent, AgentInfo> agentInfo)
    {
        var brainName = brain.gameObject.name;
        triedSendState[brainName] = true;


        if (!current_agents.ContainsKey(brainName))
        {
            current_agents[brainName] = new List<Agent>();
        }
        current_agents[brainName].Clear();
        foreach (Agent agent in agentInfo.Keys)
        {
            current_agents[brainName].Add(agent);
        }
        if (current_agents[brainName].Count() > 0)
        {
            hasSentState[brainName] = true;
            concatenatedAgentId.Clear();
            concatenatedStates.Clear();
            concatenatedRewards.Clear();
            concatenatedMemories.Clear();
            concatenatedDones.Clear();
            concatenatedActions.Clear();

            foreach (Agent agent in current_agents[brainName])
            {
                concatenatedAgentId.Add(agentInfo[agent].id);
                concatenatedStates.AddRange(agentInfo[agent].stakedVectorObservation);
                concatenatedRewards.Add(agentInfo[agent].reward);
                concatenatedMemories.AddRange(agentInfo[agent].memories.ToList());
                concatenatedDones.Add(agentInfo[agent].done);
                concatenatedActions.AddRange(agentInfo[agent].StoredVectorActions.ToList());


            }

            sMessage.brain_name = brainName;
            sMessage.agents = concatenatedAgentId;
            sMessage.states = concatenatedStates;
            sMessage.rewards = concatenatedRewards;
            sMessage.actions = concatenatedActions;
            sMessage.memories = concatenatedMemories;
            sMessage.dones = concatenatedDones;
            sMessage.maxes = concatenatedMaxes;

            sMessageString = JsonUtility.ToJson(sMessage);
            sender.Send(AppendLength(Encoding.ASCII.GetBytes(sMessageString)));
            Receive();
            int i = 0;
            foreach (resolution res in brain.brainParameters.cameraResolutions)
            {
                foreach (Agent agent in current_agents[brainName])
                {
                    sender.Send(AppendLength(TexToByteArray(agentInfo[agent].visualObservations[i])));
                    Receive();
                }
                i++;
            }


        }
        if (triedSendState.Values.All(x => x))
        {
            if (hasSentState.Values.Any(x => x))
            {
                // if all the brains listed have sent their state
                sender.Send(AppendLength(Encoding.ASCII.GetBytes("END_OF_MESSAGE:" + (academy.done ? "True" : "False"))));


                UpdateCommand();
                if (GetCommand() == ExternalCommand.STEP)
                {
                    UpdateActions();
                }
            }

            List<string> brainNames = current_agents.Keys.ToList();
            foreach (string k in brainNames)
            {
                hasSentState[k] = false;
                triedSendState[k] = false;
            }
        }

    }

    /// Listens for actions, memories, and values and sends them 
    /// to the corrensponding brains.
    public void UpdateActions()
    {
        sender.Send(Encoding.ASCII.GetBytes("STEPPING"));
        ReceiveAll();
        var agentMessage = JsonConvert.DeserializeObject<AgentMessage>(rMessage);

        foreach (Brain brain in brains)
        {
            if (brain.brainType == BrainType.External)
            {
                var brainName = brain.gameObject.name;


                for (int i = 0; i < current_agents[brainName].Count(); i++)
                {
                    Agent agent = current_agents[brainName][i];
                    if (brain.brainParameters.actionSpaceType == StateType.continuous)
                    {
                        agent.UpdateVectorAction(agentMessage.action[brainName].GetRange(i * brain.brainParameters.actionSize, brain.brainParameters.actionSize).ToArray());
                    }
                    else
                    {
                        agent.UpdateVectorAction(agentMessage.action[brainName].GetRange(i, 1).ToArray());

                    }

                    agent.UpdateMemoriesAction(agentMessage.memory[brainName].GetRange(i * brain.brainParameters.memorySize, brain.brainParameters.memorySize).ToArray());


                    agent.UpdateValueAction(agentMessage.value[brainName][i]);

                }

            }
        }
    }



}