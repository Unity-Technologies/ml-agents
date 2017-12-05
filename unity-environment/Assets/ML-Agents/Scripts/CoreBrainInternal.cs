using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

using System.Linq;

#if ENABLE_TENSORFLOW
using TensorFlow;
#endif

/// CoreBrain which decides actions using internally embedded TensorFlow model.
public class CoreBrainInternal : ScriptableObject, CoreBrain
{

    [SerializeField]
    private bool broadcast = true;

    [System.Serializable]
    private struct TensorFlowAgentPlaceholder
    {
        public enum tensorType
        {
            Integer,
            FloatingPoint}

        ;

        public string name;
        public tensorType valueType;
        public float minValue;
        public float maxValue;

    }

    ExternalCommunicator coord;

    /// Modify only in inspector : Reference to the Graph asset
    public TextAsset graphModel;
    /// Modify only in inspector : If a scope was used when training the model, specify it here
    public string graphScope;
    [SerializeField]
    ///  Modify only in inspector : If your graph takes additional inputs that are fixed you can specify them here.
    private TensorFlowAgentPlaceholder[] graphPlaceholders;
    ///  Modify only in inspector : Name of the placholder of the batch size
    public string BatchSizePlaceholderName = "batch_size";
    ///  Modify only in inspector : Name of the state placeholder
    public string StatePlacholderName = "state";
    ///  Modify only in inspector : Name of the recurrent input
    public string RecurrentInPlaceholderName = "recurrent_in";
    ///  Modify only in inspector : Name of the recurrent output
    public string RecurrentOutPlaceholderName = "recurrent_out";
    /// Modify only in inspector : Names of the observations placeholders
    public string[] ObservationPlaceholderName;
    /// Modify only in inspector : Name of the action node
    public string ActionPlaceholderName = "action";
    #if ENABLE_TENSORFLOW
    TFGraph graph;
    TFSession session;
    bool hasRecurrent;
    bool hasState;
    bool hasBatchSize;
    List<int> agentKeys;
    int currentBatchSize;
    float[,] inputState;
    List<float[,,,]> observationMatrixList;
    float[,] inputOldMemories;
    #endif

    /// Reference to the brain that uses this CoreBrainInternal
    public Brain brain;

    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    /// Loads the tensorflow graph model to generate a TFGraph object
    public void InitializeCoreBrain()
    {
#if ENABLE_TENSORFLOW
#if UNITY_ANDROID
		// This needs to ba called only once and will raise an exception if 
        // there are multiple internal brains
		try{
			TensorFlowSharp.Android.NativeBinding.Init();
		}
		catch{
			
		}
#endif
        if ((brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator == null)
        || (!broadcast))
        {
            coord = null;
        }
        else if (brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator is ExternalCommunicator)
        {
            coord = (ExternalCommunicator)brain.gameObject.transform.parent.gameObject.GetComponent<Academy>().communicator;
            coord.SubscribeBrain(brain);
        }

        if (graphModel != null)
        {

            graph = new TFGraph();

            graph.Import(graphModel.bytes);

            session = new TFSession(graph);

            if (graph[graphScope + BatchSizePlaceholderName] != null)
            {
                hasBatchSize = true;
            }
            if ((graph[graphScope + RecurrentInPlaceholderName] != null) && (graph[graphScope + RecurrentOutPlaceholderName] != null))
            {
                hasRecurrent = true;

            }
            if (graph[graphScope + StatePlacholderName] != null)
            {
                hasState = true;
            }
        }
#endif
    }


    /// Collects information from the agents and store them
    public void SendState()
    {
#if ENABLE_TENSORFLOW
        agentKeys = new List<int>(brain.agents.Keys);
        currentBatchSize = brain.agents.Count;
        if (currentBatchSize == 0)
        {

            if (coord != null)
            {
                coord.giveBrainInfo(brain);
            }
            return;
        }


        // Create the state tensor
        if (hasState)
        {
            Dictionary<int, List<float>> states = brain.CollectStates();
            inputState = new float[currentBatchSize, brain.brainParameters.stateSize];
            int i = 0;
            foreach (int k in agentKeys)
            {
                List<float> state_list = states[k];
                for (int j = 0; j < brain.brainParameters.stateSize; j++)
                {

                    inputState[i, j] = state_list[j];
                }
                i++;
            }
        }


        // Create the observation tensors
        observationMatrixList = brain.GetObservationMatrixList(agentKeys);

        // Create the recurrent tensor
        if (hasRecurrent)
        {
            Dictionary<int, float[]> old_memories = brain.CollectMemories();
            inputOldMemories = new float[currentBatchSize, brain.brainParameters.memorySize];
            int i = 0;
            foreach (int k in agentKeys)
            {
                float[] m = old_memories[k];
                for (int j = 0; j < brain.brainParameters.memorySize; j++)
                {

                    inputOldMemories[i, j] = m[j];
                }
                i++;
            }
        }


        if (coord != null)
        {
            coord.giveBrainInfo(brain);
        }
        #endif
    }


    /// Uses the stored information to run the tensorflow graph and generate 
    /// the actions.
    public void DecideAction()
    {
#if ENABLE_TENSORFLOW
        if (currentBatchSize == 0)
        {
            return;
        }

        var runner = session.GetRunner();
        runner.Fetch(graph[graphScope + ActionPlaceholderName][0]);

        if (hasBatchSize)
        {
            runner.AddInput(graph[graphScope + BatchSizePlaceholderName][0], new int[] { currentBatchSize });
        }

        foreach (TensorFlowAgentPlaceholder placeholder in graphPlaceholders)
        {
            try
            {
                if (placeholder.valueType == TensorFlowAgentPlaceholder.tensorType.FloatingPoint)
                {
                    runner.AddInput(graph[graphScope + placeholder.name][0], new float[] { Random.Range(placeholder.minValue, placeholder.maxValue) });
                }
                else if (placeholder.valueType == TensorFlowAgentPlaceholder.tensorType.Integer)
                {
                    runner.AddInput(graph[graphScope + placeholder.name][0], new int[] { Random.Range((int)placeholder.minValue, (int)placeholder.maxValue + 1) });
                }
            }
            catch
            {
                throw new UnityAgentsException(string.Format(@"One of the Tensorflow placeholder cound nout be found.
                In brain {0}, there are no {1} placeholder named {2}.",
                        brain.gameObject.name, placeholder.valueType.ToString(), graphScope + placeholder.name));
            }
        }

        // Create the state tensor
        if (hasState)
        {
            if (brain.brainParameters.stateSpaceType == StateType.discrete)
            {
                int[,] discreteInputState = new int[currentBatchSize, 1];
                for (int i = 0; i < currentBatchSize; i++)
                {
                    discreteInputState[i, 0] = (int)inputState[i, 0];
                }
                runner.AddInput(graph[graphScope + StatePlacholderName][0], discreteInputState);
            }
            else
            {
                runner.AddInput(graph[graphScope + StatePlacholderName][0], inputState);
            }
        }

        // Create the observation tensors
        for (int obs_number = 0; obs_number < brain.brainParameters.cameraResolutions.Length; obs_number++)
        {
            runner.AddInput(graph[graphScope + ObservationPlaceholderName[obs_number]][0], observationMatrixList[obs_number]);
        }

        if (hasRecurrent)
        {
            runner.AddInput(graph[graphScope + RecurrentInPlaceholderName][0], inputOldMemories);
            runner.Fetch(graph[graphScope + RecurrentOutPlaceholderName][0]);
        }
            
        TFTensor[] networkOutput;
        try
        {
            networkOutput = runner.Run();
        }
        catch (TFException e)
        {
            string errorMessage = e.Message;
            try
            {
                errorMessage = string.Format(@"The tensorflow graph needs an input for {0} of type {1}",
                    e.Message.Split(new string[]{ "Node: " }, 0)[1].Split('=')[0],
                    e.Message.Split(new string[]{ "dtype=" }, 0)[1].Split(',')[0]);
            }
            finally
            {
                throw new UnityAgentsException(errorMessage);
            }

        }

        // Create the recurrent tensor
        if (hasRecurrent)
        {
            Dictionary<int, float[]> new_memories = new Dictionary<int, float[]>();

            float[,] recurrent_tensor = networkOutput[1].GetValue() as float[,];

            int i = 0;
            foreach (int k in agentKeys)
            {
                float[] m = new float[brain.brainParameters.memorySize];
                for (int j = 0; j < brain.brainParameters.memorySize; j++)
                {
                    m[j] = recurrent_tensor[i, j];
                }
                new_memories.Add(k, m);
                i++;
            }

            brain.SendMemories(new_memories);
        }

        Dictionary<int, float[]> actions = new Dictionary<int, float[]>();

        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            float[,] output = networkOutput[0].GetValue() as float[,];
            int i = 0;
            foreach (int k in agentKeys)
            {
                float[] a = new float[brain.brainParameters.actionSize];
                for (int j = 0; j < brain.brainParameters.actionSize; j++)
                {
                    a[j] = output[i, j];
                }
                actions.Add(k, a);
                i++;
            }
        }
        else if (brain.brainParameters.actionSpaceType == StateType.discrete)
        {
            long[,] output = networkOutput[0].GetValue() as long[,];
            int i = 0;
            foreach (int k in agentKeys)
            {
                float[] a = new float[1] { (float)(output[i, 0]) };
                actions.Add(k, a);
                i++;
            }
        }

        brain.SendActions(actions);

#endif
    }

    /// Displays the parameters of the CoreBrainInternal in the Inspector 
    public void OnInspector()
    {
#if ENABLE_TENSORFLOW && UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        broadcast = EditorGUILayout.Toggle("Broadcast", broadcast);
        SerializedObject serializedBrain = new SerializedObject(this);
        GUILayout.Label("Edit the Tensorflow graph parameters here");
        SerializedProperty tfGraphModel = serializedBrain.FindProperty("graphModel");
        serializedBrain.Update();
        EditorGUILayout.ObjectField(tfGraphModel);
        serializedBrain.ApplyModifiedProperties();

        if (graphModel == null)
        {
            EditorGUILayout.HelpBox("Please provide a tensorflow graph as a bytes file.", MessageType.Error);
        }


        graphScope = EditorGUILayout.TextField("Graph Scope : ", graphScope);

        if (BatchSizePlaceholderName == "")
        {
            BatchSizePlaceholderName = "batch_size";
        }
        BatchSizePlaceholderName = EditorGUILayout.TextField("Batch Size Node Name", BatchSizePlaceholderName);
        if (StatePlacholderName == "")
        {
            StatePlacholderName = "state";
        }
        StatePlacholderName = EditorGUILayout.TextField("State Node Name", StatePlacholderName);
        if (RecurrentInPlaceholderName == "")
        {
            RecurrentInPlaceholderName = "recurrent_in";
        }
        RecurrentInPlaceholderName = EditorGUILayout.TextField("Recurrent Input Node Name", RecurrentInPlaceholderName);
        if (RecurrentOutPlaceholderName == "")
        {
            RecurrentOutPlaceholderName = "recurrent_out";
        }
        RecurrentOutPlaceholderName = EditorGUILayout.TextField("Recurrent Output Node Name", RecurrentOutPlaceholderName);

        if (brain.brainParameters.cameraResolutions != null)
        {
            if (brain.brainParameters.cameraResolutions.Count() > 0)
            {
                if (ObservationPlaceholderName == null)
                {
                    ObservationPlaceholderName = new string[brain.brainParameters.cameraResolutions.Count()];
                }
                if (ObservationPlaceholderName.Count() != brain.brainParameters.cameraResolutions.Count())
                {
                    ObservationPlaceholderName = new string[brain.brainParameters.cameraResolutions.Count()];
                }
                for (int obs_number = 0; obs_number < brain.brainParameters.cameraResolutions.Count(); obs_number++)
                {
                    if ((ObservationPlaceholderName[obs_number] == "") || (ObservationPlaceholderName[obs_number] == null))
                    {

                        ObservationPlaceholderName[obs_number] = "observation_" + obs_number;
                    }
                }
                SerializedProperty opn = serializedBrain.FindProperty("ObservationPlaceholderName");
                serializedBrain.Update();
                EditorGUILayout.PropertyField(opn, true);
                serializedBrain.ApplyModifiedProperties();
            }
        }

        if (ActionPlaceholderName == "")
        {
            ActionPlaceholderName = "action";
        }
        ActionPlaceholderName = EditorGUILayout.TextField("Action Node Name", ActionPlaceholderName);



        SerializedProperty tfPlaceholders = serializedBrain.FindProperty("graphPlaceholders");
        serializedBrain.Update();
        EditorGUILayout.PropertyField(tfPlaceholders, true);
        serializedBrain.ApplyModifiedProperties();
#endif
    }

}
