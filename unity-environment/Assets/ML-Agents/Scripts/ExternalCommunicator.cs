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

    Academy academy;

    Dictionary<string, List<int>> current_agents;

    List<Brain> brains;

    Dictionary<string, bool> hasSentState;

    Dictionary<string, Dictionary<int, float[]>> storedActions;
    Dictionary<string, Dictionary<int, float[]>> storedMemories;
    Dictionary<string, Dictionary<int, float>> storedValues;

    // For Messages
    List<float> concatenatedStates = new List<float>(1024);
    List<float> concatenatedRewards = new List<float>(32);
    List<float> concatenatedMemories = new List<float>(1024);
    List<bool> concatenatedDones = new List<bool>(32);
    List<float> concatenatedActions = new List<float>(1024);

    private int comPort;
    Socket sender;
    byte[] messageHolder;

    const int messageLength = 12000;

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
        current_agents = new Dictionary<string, List<int>>();

        hasSentState = new Dictionary<string, bool>();

        storedActions = new Dictionary<string, Dictionary<int, float[]>>();
        storedMemories = new Dictionary<string, Dictionary<int, float[]>>();
        storedValues = new Dictionary<string, Dictionary<int, float>>();
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
    public ExternalCommand GetCommand()
    {
        int location = sender.Receive(messageHolder);
        string message = Encoding.ASCII.GetString(messageHolder, 0, location);
        switch (message)
        {
            case "STEP":
                return ExternalCommand.STEP;
            case "RESET":
                return ExternalCommand.RESET;
            case "QUIT":
                return ExternalCommand.QUIT;
            default:
                return ExternalCommand.QUIT;
        }
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
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--port")
            {
                inputPort = args[i + 1];
            }
        }

        comPort = int.Parse(inputPort);
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
    public void giveBrainInfo(Brain brain)
    {
        var brainName = brain.gameObject.name;
        current_agents[brainName] = new List<int>(brain.agents.Keys);
        brain.CollectEverything();

        concatenatedStates.Clear();
        concatenatedRewards.Clear();
        concatenatedMemories.Clear();
        concatenatedDones.Clear();
        concatenatedActions.Clear();

        foreach (int id in current_agents[brainName])
        {
            concatenatedStates.AddRange(brain.currentStates[id]);
            concatenatedRewards.Add(brain.currentRewards[id]);
            concatenatedMemories.AddRange(brain.currentMemories[id].ToList());
            concatenatedDones.Add(brain.currentDones[id]);
            concatenatedActions.AddRange(brain.currentActions[id].ToList());
        }

        sMessage.brain_name = brainName;
        sMessage.agents = current_agents[brainName];
        sMessage.states = concatenatedStates;
        sMessage.rewards = concatenatedRewards;
        sMessage.actions = concatenatedActions;
        sMessage.memories = concatenatedMemories;
        sMessage.dones = concatenatedDones;

        sMessageString = JsonUtility.ToJson(sMessage);
        sender.Send(AppendLength(Encoding.ASCII.GetBytes(sMessageString)));
        Receive();
        int i = 0;
        foreach (resolution res in brain.brainParameters.cameraResolutions)
        {
            foreach (int id in current_agents[brainName])
            {
                sender.Send(AppendLength(TexToByteArray(brain.ObservationToTex(brain.currentCameras[id][i], res.width, res.height))));
                Receive();
            }
            i++;
        }

        hasSentState[brainName] = true;

        if (hasSentState.Values.All(x => x))
        {
            // if all the brains listed have sent their state
            sender.Send(Encoding.ASCII.GetBytes((academy.done ? "True" : "False")));
            List<string> brainNames = hasSentState.Keys.ToList();
            foreach (string k in brainNames)
            {
                hasSentState[k] = false;
            }
        }

    }

    /// Listens for actions, memories, and values and sends them 
    /// to the corrensponding brains.
    public void UpdateActions()
    {
        // TO MODIFY	--------------------------------------------
        sender.Send(Encoding.ASCII.GetBytes("STEPPING"));
        Receive();
        var agentMessage = JsonConvert.DeserializeObject<AgentMessage>(rMessage);

        foreach (Brain brain in brains)
        {
            if (brain.brainType == BrainType.External)
            {
                var brainName = brain.gameObject.name;

                var actionDict = new Dictionary<int, float[]>();
                var memoryDict = new Dictionary<int, float[]>();
                var valueDict = new Dictionary<int, float>();

                for (int i = 0; i < current_agents[brainName].Count; i++)
                {
                    if (brain.brainParameters.actionSpaceType == StateType.continuous)
                    {
                        actionDict.Add(current_agents[brainName][i],
                            agentMessage.action[brainName].GetRange(i * brain.brainParameters.actionSize, brain.brainParameters.actionSize).ToArray());
                    }
                    else
                    {
                        actionDict.Add(current_agents[brainName][i],
                            agentMessage.action[brainName].GetRange(i, 1).ToArray());
                    }

                    memoryDict.Add(current_agents[brainName][i],
    agentMessage.memory[brainName].GetRange(i * brain.brainParameters.memorySize, brain.brainParameters.memorySize).ToArray());
                    
                    valueDict.Add(current_agents[brainName][i],
    agentMessage.value[brainName][i]);

                }
                storedActions[brainName] = actionDict;
                storedMemories[brainName] = memoryDict;
                storedValues[brainName] = valueDict;
            }
        }
    }

    /// Returns the actions corrensponding to the brain called brainName that 
    /// were received throught the socket.
    public Dictionary<int, float[]> GetDecidedAction(string brainName)
    {
        return storedActions[brainName];
    }

    /// Returns the memories corrensponding to the brain called brainName that
    ///  were received throught the socket.
    public Dictionary<int, float[]> GetMemories(string brainName)
    {
        return storedMemories[brainName];
    }

    /// Returns the values corrensponding to the brain called brainName that 
    /// were received throught the socket.
    public Dictionary<int, float> GetValues(string brainName)
    {
        return storedValues[brainName];
    }

}
