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

    IEnumerable<AgentInfo> agentInfos;
    Dictionary<string, List<AgentAction>> agentActions;

    List<Brain> brains;
    Dictionary<string, bool> hasSentState;
    Dictionary<string, bool> triedSendState;

    const int messageLength = 12000;
    const int defaultNumAgents = 32;
    const int defaultNumObservations = 32;


    int comPort;
    int randomSeed;
    Socket sender;
    byte[] messageHolder;
    byte[] lengthHolder;

    StreamWriter logWriter;
    string logPath;

    const string _version_ = "API-3";

    /// Placeholder for state information to send.
    [System.Serializable]
    [HideInInspector]
    public struct StepMessage
    {
        public string brain_name;
        public List<int> agents;
        public List<float> vectorObservations;
        public List<float> rewards;
        public List<float> previousVectorActions;
        public List<string> previousTextActions;
        public List<float> memories;
        public List<string> textObservations;
        public List<bool> dones;
        public List<bool> maxes;
    }

    StepMessage sMessage;
    string sMessageString;

    AgentMessage rMessage;
    StringBuilder rMessageString = new StringBuilder(messageLength);

    /// Placeholder for returned message.
    struct AgentMessage
    {
        public Dictionary<string, List<float>> vector_action { get; set; }
        public Dictionary<string, List<float>> memory { get; set; }
        public Dictionary<string, List<string>> text_action { get; set; }
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
        agentInfos = new List<AgentInfo>();
        agentActions = new Dictionary<string, List<AgentAction>>();

        hasSentState = new Dictionary<string, bool>();
        triedSendState = new Dictionary<string, bool>();

    }

    /// Adds the brain to the list of brains which have already decided their
    /// actions.
    public void SubscribeBrain(Brain brain)
    {
        brains.Add(brain);
        triedSendState[brain.gameObject.name] = false;
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
        accParamerters.apiNumber = _version_;
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
        sMessage.agents = new List<int>(defaultNumAgents);
        sMessage.vectorObservations = new List<float>(defaultNumAgents * defaultNumObservations);
        sMessage.rewards = new List<float>(defaultNumAgents);
        sMessage.memories = new List<float>(defaultNumAgents * defaultNumObservations);
        sMessage.dones = new List<bool>(defaultNumAgents);
        sMessage.previousVectorActions = new List<float>(defaultNumAgents * defaultNumObservations);
        sMessage.previousTextActions = new List<string>(defaultNumAgents);
        sMessage.maxes = new List<bool>(defaultNumAgents);
        sMessage.textObservations = new List<string>(defaultNumAgents);

        // Initialize the list of brains the Communicator must listen to
        // Issue : This assumes all brains are broadcasting.
        foreach (string k in accParamerters.brainNames)
        {
            agentActions[k] = new List<AgentAction>(defaultNumAgents);
            hasSentState[k] = false;
            triedSendState[k] = false;
        }

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
        var resetParams = JsonConvert.DeserializeObject<ResetParametersMessage>(rMessageString.ToString());
        academy.SetIsInference(!resetParams.train_model);
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
        rMessageString.Clear();
        rMessageString.Append(Encoding.ASCII.GetString(messageHolder, 0, location));
    }

    /// Receives a message and can reconstruct a message if was too long
    private void ReceiveAll()
    {
        sender.Receive(lengthHolder);
        int totalLength = System.BitConverter.ToInt32(lengthHolder, 0);
        int location = 0;
        rMessageString.Clear();
        while (location != totalLength)
        {
            int fragment = sender.Receive(messageHolder);
            location += fragment;
            rMessageString.Append(Encoding.ASCII.GetString(messageHolder, 0, fragment));
        }
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
    public void GiveBrainInfo(Brain brain, Dictionary<AgentInfo, AgentAction> agentRequest)
    {
        var brainName = brain.gameObject.name;
        triedSendState[brainName] = true;


        agentActions[brainName].Clear();
        agentInfos = agentRequest.Keys;
        foreach (AgentInfo info in agentInfos)
        {
            agentActions[brainName].Add(agentRequest[info]);
        }
        if (agentActions[brainName].Count() > 0)
        {
            hasSentState[brainName] = true;
            sMessage.brain_name = brainName;
            sMessage.agents.Clear();
            sMessage.vectorObservations.Clear();
            sMessage.rewards.Clear();
            sMessage.memories.Clear();
            sMessage.dones.Clear();
            sMessage.previousVectorActions.Clear();
            sMessage.previousTextActions.Clear();
            sMessage.maxes.Clear();
            sMessage.textObservations.Clear();

            int memorySize = 0;
            foreach (AgentInfo info in agentInfos)
            {
                memorySize = Mathf.Max(info.memories.Count, memorySize);
            }

            foreach (AgentInfo info in agentInfos)
            {
                sMessage.agents.Add(info.id);
                sMessage.vectorObservations.AddRange(info.stackedVectorObservation);
                sMessage.rewards.Add(info.reward);
                sMessage.memories.AddRange(info.memories);
                for (int j = 0; j < memorySize - info.memories.Count; j++)
                {
                    sMessage.memories.Add(0f);
                }
                sMessage.dones.Add(info.done);
                sMessage.previousVectorActions.AddRange(info.storedVectorActions);
                sMessage.previousTextActions.Add(info.storedTextActions);
                sMessage.maxes.Add(info.maxStepReached);
                sMessage.textObservations.Add(info.textObservation);

            }



            sMessageString = JsonUtility.ToJson(sMessage);
            sender.Send(AppendLength(Encoding.ASCII.GetBytes(sMessageString)));
            Receive();
            int i = 0;
            foreach (resolution res in brain.brainParameters.cameraResolutions)
            {
                foreach (AgentInfo info in agentInfos)
                {
                    sender.Send(AppendLength(TexToByteArray(info.visualObservations[i])));
                    Receive();
                }
                i++;
            }


        }
        if (triedSendState.Values.All(x => x))
        {
            if (hasSentState.Values.Any(x => x) || academy.IsDone())
            {
                // if all the brains listed have sent their state
                sender.Send(AppendLength(Encoding.ASCII.GetBytes("END_OF_MESSAGE:" + (academy.IsDone() ? "True" : "False"))));


                UpdateCommand();
                if (GetCommand() == ExternalCommand.STEP)
                {
                    UpdateActions();
                }
            }

            foreach (string k in agentActions.Keys)
            {
                hasSentState[k] = false;
                triedSendState[k] = false;
            }
        }

    }

    public Dictionary<string, bool> GetHasTried()
    {
        return triedSendState;
    }

    public Dictionary<string, bool> GetSent()
    {
        return hasSentState;
    }

    /// Listens for actions, memories, and values and sends them 
    /// to the corrensponding brains.
    public void UpdateActions()
    {
        sender.Send(Encoding.ASCII.GetBytes("STEPPING"));
        ReceiveAll();
        rMessage = JsonConvert.DeserializeObject<AgentMessage>(rMessageString.ToString());

        foreach (Brain brain in brains)
        {
            if (brain.brainType == BrainType.External)
            {
                var brainName = brain.gameObject.name;

                if (agentActions[brainName].Count() == 0)
                {
                    continue;
                }
                var memorySize = rMessage.memory[brainName].Count() / agentActions[brainName].Count();

                for (int i = 0; i < agentActions[brainName].Count(); i++)
                {
                    if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
                    {
                        agentActions[brainName][i].vectorActions =
                            rMessage.vector_action[brainName].GetRange(
                                i * brain.brainParameters.vectorActionSize, brain.brainParameters.vectorActionSize).ToArray();
                    }
                    else
                    {
                        agentActions[brainName][i].vectorActions =
                            rMessage.vector_action[brainName].GetRange(i, 1).ToArray();

                    }

                    agentActions[brainName][i].memories =
                        rMessage.memory[brainName].GetRange(i * memorySize, memorySize);

                    if (rMessage.text_action[brainName].Count > 0)
                    {
                        agentActions[brainName][i].textActions =
                                                      rMessage.text_action[brainName][i];
                    }
                }

            }
        }
    }
}
