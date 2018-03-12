# CoreBrain Interface

The behavior of a Brain object depends on its **Brain Type** setting. Each of the supported types of Brain implements the CoreBrain Interface. You can implement your own CoreBrain if none of the four included types do exactly what you want.

The CoreBrain interface defines the following methods:

    void SetBrain(Brain b);
    void InitializeCoreBrain();
    void DecideAction();
    void OnInspector();

Note that the name of your implementation must start with "CoreBrain" in order to add it to the list of Brain types in the Brain Inspector window. See [Adding a new Brain Type to the Brain Inspector](#adding-a-new-brain-type-to-the-brain-inspector).

## SetBrain

Use the `SetBrain()` function to store a reference to the parent Brain instance. `SetBrain()` is called before any of the other runtime CoreBrain functions, so you can use this Brain reference to access important properties of the parent Brain.

    private Brain brain;
    public void SetBrain(Brain b)
    {
        brain = b;
    }

## InitializeCoreBrain

Use `InitializeCoreBrain()` to initialize your CoreBrain instance at runtime. Since `SetBrain()` has already been called, you can access the parent Brain properties. This function is also a good place to connect your brain to the ExternalCommunicator, if you want the CoreBrain implementation to communicate with an external process:

    private ExternalCommunicator extComms;
    public void InitializeCoreBrain(Communicator communicator)
    {
        actionValues = new float[brain.brainParameters.actionSize];
        agentActions  = new Dictionary<int, float[]>();
    
        extComms = communicator as ExternalCommunicator;
        if(extComms != null)
        {
            extComms.SubscribeBrain(brain);
        }
    }


## DecideAction

Use `DecideAction()` to determine the actions of any agents using this brain. The parent Brain passes a dictionary containing each Agent object and its corresponding AgentInfo struct. The AgentInfo struct provides all of the  agent's observations and rewards.

For each agent, you must construct a `float[]` array containing the action vector elements and add this array to a Dictionary using the same integer key used by that agent in `Brain.agents`. Send this agent-action dictionary to the Brain using `Brain.SendAction()`.

    public void DecideAction()
    {
        float[] actionValues = new float[brain.brainParameters.actionSize];
        for(int i = 0; i < actionValues.Length; i++)
        {
            // Set actionValues[i]...
        }
        var agentActions = new Dictionary<int, float[]>();
        foreach (KeyValuePair<int, Agent> idAgent in brain.agents)
        {
            agentActions.Add(idAgent.Key, actionValues);
        }
        brain.SendActions(agentActions);
    }

Of course, _how_ you decide an agent's actions is a key implementation detail for a CoreBrain. For example, the CoreBrainPlayer, which maps key commands to action values, simply checks for key presses using the `Input.GetKey()` function and sets the mapped element of the action vector to the corresponding, preset value. CoreBrainPlayer does not need to use any observations or memories of the agent and, thus, is very simple.

In contrast, CoreBrainInternal feeds an agent's observations and other variables collected by the `SendState()` function into the TensorFlow data graph and then applies the output vector from the trained neural network to the agent's action vector.

To support the ExternalCommunicator broadcast function, you must send the `Brain.BrainInfo` object using the `ExternalCommmunicator.giveBrainInfo()` function. In fact if all you need to do is send the agents' observations to an external process, you can simply call this function:

    public void DecideAction()
    {
        // Assumes extComms has been set by InitializeCoreBrain() function
        if (extComms != null)
        {
            extComms.giveBrainInfo(brain);
        }
    }

The ExternalCommunicator class takes care of collecting each agent's observations and sending them to the process.

## OnInspector

Use `OnInspector()` to implement a Unity property inspector for your CoreBrain implementation. If you do not provide an implementation, users of your CoreBrain will not be able to set any its fields or properties in the Unity Editor Inspector window. See [Extending the editor](https://docs.unity3d.com/Manual/ExtendingTheEditor.html) and [EditorGUI](https://docs.unity3d.com/ScriptReference/EditorGUI.html) for more information about creating custom Inspector controls.

## Adding a new Brain Type to the Brain Inspector

For your CoreBrain implementation to appear in the list of Brain Types, you must add an entry to the Brain class' BrainType enum, which is defined in Brain.cs:

    public enum BrainType
    {
        Player,
        Heuristic,
        External,
        Internal
    }

When the Brain creates an instance of your CoreBrain, it adds the enum name to the string, "CoreBrain". Thus, the class name for the Internal brain is `CoreBrainInternal`. If you created a class named, `CoreBrainFuzzyLogic`, you would add an enum named, "FuzzyLogic", to the BrainType enum.

<!--
## Example CoreBrain implementation

Once you have determined that the existing CoreBrain implementations do not fill your needs, you can implement your own. Use `SendState()` to collect the observations from your agents and store them for use in `DecideAction()`.
-->