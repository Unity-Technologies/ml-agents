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
    [Tooltip("If checked, the brain will broadcast states and actions to Python.")]
    #pragma warning disable
    private bool broadcast = true;
    #pragma warning restore

    [System.Serializable]
    private struct TensorFlowAgentPlaceholder
    {
        public enum tensorType
        {
            Integer,
            FloatingPoint
        }

        ;

        public string name;
        public tensorType valueType;
        public float minValue;
        public float maxValue;

    }

    ExternalCommunicator coord;

    [Tooltip("This must be the bytes file corresponding to the pretrained Tensorflow graph.")]
    /// Modify only in inspector : Reference to the Graph asset
    public TextAsset graphModel;

    /// Modify only in inspector : If a scope was used when training the model, specify it here
    public string graphScope;
    [SerializeField]
    [Tooltip("If your graph takes additional inputs that are fixed (example: noise level) you can specify them here.")]
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
    float[,] inputState;
    List<float[,,,]> observationMatrixList;
    float[,] inputOldMemories;
    List<Texture2D> texturesHolder;
    int memorySize;
#endif

    /// Reference to the brain that uses this CoreBrainInternal
    public Brain brain;

    /// Create the reference to the brain
    public void SetBrain(Brain b)
    {
        brain = b;
    }

    /// Loads the tensorflow graph model to generate a TFGraph object
    public void InitializeCoreBrain(Communicator communicator)
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
        if ((communicator == null)
        || (!broadcast))
        {
            coord = null;
        }
        else if (communicator is ExternalCommunicator)
        {
            coord = (ExternalCommunicator)communicator;
            coord.SubscribeBrain(brain);
        }

        if (graphModel != null)
        {

            graph = new TFGraph();

            graph.Import(graphModel.bytes);

            session = new TFSession(graph);

            if ((graphScope.Length > 1) && (graphScope[graphScope.Length - 1] != '/'))
            {
                graphScope = graphScope + '/';
            }

            if (graph[graphScope + BatchSizePlaceholderName] != null)
            {
                hasBatchSize = true;
            }
            if ((graph[graphScope + RecurrentInPlaceholderName] != null) && (graph[graphScope + RecurrentOutPlaceholderName] != null))
            {
                hasRecurrent = true;
                var runner = session.GetRunner();
                runner.Fetch(graph[graphScope + "memory_size"][0]);
                var networkOutput = runner.Run()[0].GetValue();
                memorySize = (int)networkOutput;

            }
            if (graph[graphScope + StatePlacholderName] != null)
            {
                hasState = true;
            }
        }
        observationMatrixList = new List<float[,,,]>();
        texturesHolder = new List<Texture2D>();
#endif
    }



    /// Uses the stored information to run the tensorflow graph and generate 
    /// the actions.
    public void DecideAction(Dictionary<Agent, AgentInfo> agentInfo)
    {
#if ENABLE_TENSORFLOW
		if (coord != null)
		{
			coord.GiveBrainInfo(brain, agentInfo);
		}
        int currentBatchSize = agentInfo.Count();
        List<Agent> agentList = agentInfo.Keys.ToList();
        if (currentBatchSize == 0)
        {
            return;
        }


        // Create the state tensor
        if (hasState)
        {
            int stateLength = 1;
            if (brain.brainParameters.vectorActionSpaceType == StateType.continuous)
            {
                stateLength = brain.brainParameters.vectorObservationSize;
            }
            inputState = new float[currentBatchSize, stateLength * brain.brainParameters.numStackedVectorObservations];

            var i = 0;
            foreach (Agent agent in agentList)
            {
                List<float> state_list = agentInfo[agent].stackedVectorObservation;
                for (int j = 0; j < brain.brainParameters.vectorObservationSize * brain.brainParameters.numStackedVectorObservations; j++)
                {
                    inputState[i, j] = state_list[j];
                }
                i++;
            }
        }



        observationMatrixList.Clear();
        for (int observationIndex = 0; observationIndex < brain.brainParameters.cameraResolutions.Count(); observationIndex++){
            texturesHolder.Clear();
            foreach (Agent agent in agentList){
                texturesHolder.Add(agentInfo[agent].visualObservations[observationIndex]);
            }
            observationMatrixList.Add(
                BatchVisualObservations(texturesHolder, brain.brainParameters.cameraResolutions[observationIndex].blackAndWhite));

        }

        // Create the recurrent tensor
        if (hasRecurrent)
        {
            // Need to have variable memory size
            inputOldMemories = new float[currentBatchSize, memorySize];
            var i = 0;
            foreach (Agent agent in agentList)
            {
                float[] m = agentInfo[agent].memories.ToArray();
                for (int j = 0; j < m.Count(); j++)
                {

                    inputOldMemories[i, j] = m[j];

                }
                i++;
            }
        }


        var runner = session.GetRunner();
        try
        {
            runner.Fetch(graph[graphScope + ActionPlaceholderName][0]);
        }
        catch
        {
            throw new UnityAgentsException(string.Format(@"The node {0} could not be found. Please make sure the graphScope {1} is correct",
                     graphScope + ActionPlaceholderName, graphScope));
        }

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
            if (brain.brainParameters.vectorObservationSpaceType == StateType.discrete)
            {
                var discreteInputState = new int[currentBatchSize, 1];
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

        //// Create the observation tensors
        for (int obs_number = 0; obs_number < brain.brainParameters.cameraResolutions.Length; obs_number++)
        {
            runner.AddInput(graph[graphScope + ObservationPlaceholderName[obs_number]][0], observationMatrixList[obs_number]);
        }

        if (hasRecurrent)
        {
            runner.AddInput(graph[graphScope + "sequence_length"][0], 1);
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
                    e.Message.Split(new string[] { "Node: " }, 0)[1].Split('=')[0],
                    e.Message.Split(new string[] { "dtype=" }, 0)[1].Split(',')[0]);
            }
            finally
            {
                throw new UnityAgentsException(errorMessage);
            }

        }

        // Create the recurrent tensor
        if (hasRecurrent)
        {
            float[,] recurrent_tensor = networkOutput[1].GetValue() as float[,];

            var i = 0;
            foreach (Agent agent in agentList)
            {
                var m = new float[memorySize];
                for (int j = 0; j < memorySize; j++)
                {
                    m[j] = recurrent_tensor[i, j];
                }
                agent.UpdateMemoriesAction(m.ToList());
                i++;
            }

        }

        if (brain.brainParameters.vectorActionSpaceType == StateType.continuous)
        {
            var output = networkOutput[0].GetValue() as float[,];
            var i = 0;
            foreach (Agent agent in agentList)
            {
                var a = new float[brain.brainParameters.vectorActionSize];
                for (int j = 0; j < brain.brainParameters.vectorActionSize; j++)
                {
                    a[j] = output[i, j];
                }
                agent.UpdateVectorAction(a);
                i++;
            }
        }
        else if (brain.brainParameters.vectorActionSpaceType == StateType.discrete)
        {
            long[,] output = networkOutput[0].GetValue() as long[,];
            var i = 0;
            foreach (Agent agent in agentList)
            {
                var a = new float[1] { (float)(output[i, 0]) };
                agent.UpdateVectorAction(a);
                i++;
            }
        }




#else
        if (agentInfo.Count > 0)
        {
            throw new UnityAgentsException(string.Format(@"The brain {0} was set to Internal but the Tensorflow 
                        library is not present in the Unity project.",
                        brain.gameObject.name));
        }
#endif
    }

    /// Displays the parameters of the CoreBrainInternal in the Inspector 
    public void OnInspector()
    {
#if ENABLE_TENSORFLOW && UNITY_EDITOR
        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        broadcast = EditorGUILayout.Toggle(new GUIContent("Broadcast",
                      "If checked, the brain will broadcast states and actions to Python."), broadcast);

        var serializedBrain = new SerializedObject(this);
        GUILayout.Label("Edit the Tensorflow graph parameters here");
        var tfGraphModel = serializedBrain.FindProperty("graphModel");
        serializedBrain.Update();
        EditorGUILayout.ObjectField(tfGraphModel);
        serializedBrain.ApplyModifiedProperties();

        if (graphModel == null)
        {
            EditorGUILayout.HelpBox("Please provide a tensorflow graph as a bytes file.", MessageType.Error);
        }


        graphScope = EditorGUILayout.TextField(new GUIContent("Graph Scope", "If you set a scope while training your tensorflow model, " +
                           "all your placeholder name will have a prefix. You must specify that prefix here."), graphScope);

        if (BatchSizePlaceholderName == "")
        {
            BatchSizePlaceholderName = "batch_size";
        }
        BatchSizePlaceholderName = EditorGUILayout.TextField(new GUIContent("Batch Size Node Name", "If the batch size is one of " +
                            "the inputs of your graph, you must specify the name if the placeholder here."), BatchSizePlaceholderName);
        if (StatePlacholderName == "")
        {
            StatePlacholderName = "state";
        }
        StatePlacholderName = EditorGUILayout.TextField(new GUIContent("State Node Name", "If your graph uses the state as an input, " +
                            "you must specify the name if the placeholder here."), StatePlacholderName);
        if (RecurrentInPlaceholderName == "")
        {
            RecurrentInPlaceholderName = "recurrent_in";
        }
        RecurrentInPlaceholderName = EditorGUILayout.TextField(new GUIContent("Recurrent Input Node Name", "If your graph uses a " +
                          "recurrent input / memory as input and outputs new recurrent input / memory, " +
                          "you must specify the name if the input placeholder here."), RecurrentInPlaceholderName);
        if (RecurrentOutPlaceholderName == "")
        {
            RecurrentOutPlaceholderName = "recurrent_out";
        }
        RecurrentOutPlaceholderName = EditorGUILayout.TextField(new GUIContent("Recurrent Output Node Name", " If your graph uses a " +
                           "recurrent input / memory as input and outputs new recurrent input / memory, you must specify the name if " +
                           "the output placeholder here."), RecurrentOutPlaceholderName);

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

                        ObservationPlaceholderName[obs_number] = "visual_observation_" + obs_number;
                    }
                }
                var opn = serializedBrain.FindProperty("ObservationPlaceholderName");
                serializedBrain.Update();
                EditorGUILayout.PropertyField(opn, true);
                serializedBrain.ApplyModifiedProperties();
            }
        }

        if (ActionPlaceholderName == "")
        {
            ActionPlaceholderName = "action";
        }
        ActionPlaceholderName = EditorGUILayout.TextField(new GUIContent("Action Node Name", "Specify the name of the " +
                         "placeholder corresponding to the actions of the brain in your graph. If the action space type is " +
                         "continuous, the output must be a one dimensional tensor of float of length Action Space Size, " +
                         "if the action space type is discrete, the output must be a one dimensional tensor of int " +
                         "of length 1."), ActionPlaceholderName);



        var tfPlaceholders = serializedBrain.FindProperty("graphPlaceholders");
        serializedBrain.Update();
        EditorGUILayout.PropertyField(tfPlaceholders, true);
        serializedBrain.ApplyModifiedProperties();
#endif
#if !ENABLE_TENSORFLOW && UNITY_EDITOR
        EditorGUILayout.HelpBox (@"You need to install the TensorflowSharp plugin in order to use the internal brain.", MessageType.Error);
#endif
    }

    /// Contains logic to convert the agent's cameras into observation list
    ///  (as list of float arrays)
    public static float[,,,] BatchVisualObservations(List<Texture2D> textures, bool BlackAndWhite)
    {
        int batchSize = textures.Count();
        int width = textures[0].width;
        int height = textures[0].height;
        int pixels = 0;
        if (BlackAndWhite)
            pixels = 1;
        else
            pixels = 3;
        float[,,,] result = new float[batchSize, width, height, pixels];

        for (int b = 0; b < batchSize; b++)
        {
            Color32[] cc = textures[b].GetPixels32();
            for (int w = 0; w < width; w++)
            {
                for (int h = 0; h < height; h++)
                {
                    Color32 currentPixel = cc[h * width + w];
                    if (!BlackAndWhite)
                    {
                        result[b, textures[b].height - h - 1, w, 0] = currentPixel.r;
                        result[b, textures[b].height - h - 1, w, 1] = currentPixel.g;
                        result[b, textures[b].height - h - 1, w, 2] = currentPixel.b;
                    }
                    else
                    {
                        result[b, textures[b].height - h - 1, w, 0] = (currentPixel.r + currentPixel.g + currentPixel.b) / 3;
                    }
                }
            }
        }
        return result;
    }

}
