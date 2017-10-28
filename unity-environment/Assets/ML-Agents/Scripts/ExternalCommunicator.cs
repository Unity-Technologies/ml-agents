using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Newtonsoft.Json;
using System.Linq;
using System.Net.Sockets;
using System.Text;


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

    private int comPort;
    Socket sender;
    byte[] messageHolder;

    const int messageLength = 12000;

    private class StepMessage
    {
        public string brain_name { get; set; }
        public List<int> agents { get; set; }
        public List<float> states { get; set; }
        public List<float> rewards { get; set; }
        public List<float> actions { get; set; }
        public List<float> memories { get; set; }
        public List<bool> dones { get; set; }
    }

    private class AgentMessage
    {
        public Dictionary<string, List<float>> action { get; set; }
        public Dictionary<string, List<float>> memory { get; set; }
        public Dictionary<string, List<float>> value { get; set; }

    }

    private class ResetParametersMessage
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

    /// Contains the logic for the initializtation of the socket.
    public void InitializeCommunicator()
    {
        try
        {
            ReadArgs();
        }
        catch
        {
            throw new UnityAgentsException("One of the brains was set isExternal" +
                                           " but Unity was unable to read the" +
                                           " arguments passed at launch");
        }

        messageHolder = new byte[messageLength];

        // Create a TCP/IP  socket.  
        sender = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        sender.Connect("localhost", comPort);

        AcademyParameters accParamerters = new AcademyParameters();
        accParamerters.brainParameters = new List<BrainParameters>();
        accParamerters.brainNames = new List<string>();
        foreach (Brain b in brains)
        {
            accParamerters.brainParameters.Add(b.brainParameters);
            accParamerters.brainNames.Add(b.gameObject.name);
        }
        accParamerters.AcademyName = academy.gameObject.name;
        accParamerters.resetParameters = academy.resetParameters;

        SendParameters(accParamerters);
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
        ResetParametersMessage resetParams = JsonConvert.DeserializeObject<ResetParametersMessage>(Receive());
        if (academy.isInference != !resetParams.train_model)
        {
            academy.windowResize = true;
        }
        academy.isInference = !resetParams.train_model;
        return resetParams.parameters;
    }


    /// Used to read Python-provided environment parameters
    private void ReadArgs()
    {
        string[] args = System.Environment.GetCommandLineArgs();
        string inputPort = "";
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
        Receive();
    }

    /// Receives messages from external agent
    private string Receive()
    {
        int location = sender.Receive(messageHolder);
        string message = Encoding.ASCII.GetString(messageHolder, 0, location);
        return message;
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

    /// Collects the information from the brains and sends it accross the socket
    public void giveBrainInfo(Brain brain)
    {
        string brainName = brain.gameObject.name;
        current_agents[brainName] = new List<int>(brain.agents.Keys);
        List<float> concatenatedStates = new List<float>();
        List<float> concatenatedRewards = new List<float>();
        List<float> concatenatedMemories = new List<float>();
        List<bool> concatenatedDones = new List<bool>();
        Dictionary<int, List<Camera>> collectedObservations = brain.CollectObservations();
        Dictionary<int, List<float>> collectedStates = brain.CollectStates();
        Dictionary<int, float> collectedRewards = brain.CollectRewards();
        Dictionary<int, float[]> collectedMemories = brain.CollectMemories();
        Dictionary<int, bool> collectedDones = brain.CollectDones();

        foreach (int id in current_agents[brainName])
        {
            concatenatedStates = concatenatedStates.Concat(collectedStates[id]).ToList();
            concatenatedRewards.Add(collectedRewards[id]);
            concatenatedMemories = concatenatedMemories.Concat(collectedMemories[id].ToList()).ToList();
            concatenatedDones.Add(collectedDones[id]);
        }
        StepMessage message = new StepMessage()
        {
            brain_name = brainName,
            agents = current_agents[brainName],
            states = concatenatedStates,
            rewards = concatenatedRewards,
            //actions = actionDict,
            memories = concatenatedMemories,
            dones = concatenatedDones
        };
        string envMessage = JsonConvert.SerializeObject(message, Formatting.Indented);
        sender.Send(Encoding.ASCII.GetBytes(envMessage));
        Receive();
        int i = 0;
        foreach (resolution res in brain.brainParameters.cameraResolutions)
        {
            foreach (int id in current_agents[brainName])
            {
                sender.Send(TexToByteArray(brain.ObservationToTex(collectedObservations[id][i], res.width, res.height)));
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
        string a = Receive();
        AgentMessage agentMessage = JsonConvert.DeserializeObject<AgentMessage>(a);

        foreach (Brain brain in brains)
        {
            string brainName = brain.gameObject.name;

            Dictionary<int, float[]> actionDict = new Dictionary<int, float[]>();
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
            }
            storedActions[brainName] = actionDict;

            Dictionary<int, float[]> memoryDict = new Dictionary<int, float[]>();
            for (int i = 0; i < current_agents[brainName].Count; i++)
            {
                memoryDict.Add(current_agents[brainName][i],
                    agentMessage.memory[brainName].GetRange(i * brain.brainParameters.memorySize, brain.brainParameters.memorySize).ToArray());
            }
            storedMemories[brainName] = memoryDict;

            Dictionary<int, float> valueDict = new Dictionary<int, float>();
            for (int i = 0; i < current_agents[brainName].Count; i++)
            {
                valueDict.Add(current_agents[brainName][i],
                    agentMessage.value[brainName][i]);
            }
            storedValues[brainName] = valueDict;

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
