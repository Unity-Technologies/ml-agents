using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
#if ENABLE_TENSORFLOW
using TensorFlow;

#endif

namespace MLAgents
{
    public class LearnedBrain : Brain
    {
//        [System.Serializable]
//        private struct TensorFlowAgentPlaceholder
//        {
//            public enum tensorType
//            {
//                Integer,
//                FloatingPoint
//            };
//
//            public string name;
//            public tensorType valueType;
//            public float minValue;
//            public float maxValue;
//
//        }

        [Tooltip("This must be the bytes file corresponding to the pretrained TensorFlow graph.")]
        /// Modify only in inspector : Reference to the Graph asset
        public TextAsset graphModel;

        /// Modify only in inspector : If a scope was used when training the model, specify it here
        public string graphScope;

        [SerializeField]
        [Tooltip(
            "If your graph takes additional inputs that are fixed (example: noise level) you can specify them here.")]
        ///  Modify only in inspector : If your graph takes additional inputs that are fixed you can specify them here.
//        private TensorFlowAgentPlaceholder[] graphPlaceholders;
        ///  Modify only in inspector : Name of the placholder of the batch size
        public string BatchSizePlaceholderName = "batch_size";

        ///  Modify only in inspector : Name of the state placeholder
        public string VectorObservationPlacholderName = "vector_observation";

        ///  Modify only in inspector : Name of the recurrent input
        public string RecurrentInPlaceholderName = "recurrent_in";

        ///  Modify only in inspector : Name of the recurrent output
        public string RecurrentOutPlaceholderName = "recurrent_out";

        /// Modify only in inspector : Names of the observations placeholders
        public string[] VisualObservationPlaceholderName;

        /// Modify only in inspector : Name of the action node
        public string ActionPlaceholderName = "action";

        /// Modify only in inspector : Name of the previous action node
        public string PreviousActionPlaceholderName = "prev_action";
#if ENABLE_TENSORFLOW
        TFGraph graph;
        TFSession session;
        bool hasRecurrent;
        bool hasState;
        bool hasBatchSize;
        bool hasPrevAction;
        float[,] inputState;
        int[] inputPrevAction;
        List<float[,,,]> observationMatrixList;
        float[,] inputOldMemories;
        List<Texture2D> texturesHolder;
        int memorySize;
#endif


        public override void InitializeBrain(Academy aca, MLAgents.Batcher batcher, bool external)
        {
            base.InitializeBrain(aca, batcher, external);

            InitializeGraph();
        }

        public void InitializeGraph()
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

            if (graphModel != null)
            {
                graph = new TFGraph();

                graph.Import(graphModel.bytes);

                session = new TFSession(graph);

                // TODO: Make this a loop over a dynamic set of graph inputs

                if ((graphScope.Length > 1) && (graphScope[graphScope.Length - 1] != '/'))
                {
                    graphScope = graphScope + '/';
                }

                if (graph[graphScope + BatchSizePlaceholderName] != null)
                {
                    hasBatchSize = true;
                }

                if ((graph[graphScope + RecurrentInPlaceholderName] != null) &&
                    (graph[graphScope + RecurrentOutPlaceholderName] != null))
                {
                    hasRecurrent = true;
                    var runner = session.GetRunner();
                    runner.Fetch(graph[graphScope + "memory_size"][0]);
                    var networkOutput = runner.Run()[0].GetValue();
                    memorySize = (int) networkOutput;
                }

                if (graph[graphScope + VectorObservationPlacholderName] != null)
                {
                    hasState = true;
                }

                if (graph[graphScope + PreviousActionPlaceholderName] != null)
                {
                    hasPrevAction = true;
                }
            }

            observationMatrixList = new List<float[,,,]>();
            texturesHolder = new List<Texture2D>();
#endif
        }

        protected override void DecideAction()
        {
#if ENABLE_TENSORFLOW
            if (brainBatcher != null)
            {
                brainBatcher.SendBrainInfo(this.name, agentInfo);
            }

            if (isExternal)
            {
                agentInfo.Clear();
                return;
            }

            int currentBatchSize = agentInfo.Count;
            var agentList = agentInfo.Keys;
            if (currentBatchSize == 0)
            {
                return;
            }


            // Create the state tensor
            if (hasState)
            {
                int stateLength = 1;
                if (brainParameters.vectorObservationSpaceType == SpaceType.continuous)
                {
                    stateLength = brainParameters.vectorObservationSize;
                }

                inputState =
                    new float[currentBatchSize,
                        stateLength * brainParameters.numStackedVectorObservations];

                var i = 0;
                foreach (Agent agent in agentList)
                {
                    List<float> state_list = agentInfo[agent].stackedVectorObservation;
                    for (int j =
                            0;
                        j < stateLength * brainParameters.numStackedVectorObservations;
                        j++)
                    {
                        inputState[i, j] = state_list[j];
                    }

                    i++;
                }
            }

            // Create the state tensor
            if (hasPrevAction)
            {
                inputPrevAction = new int[currentBatchSize];
                var i = 0;
                foreach (Agent agent in agentList)
                {
                    float[] action_list = agentInfo[agent].storedVectorActions;
                    inputPrevAction[i] = Mathf.FloorToInt(action_list[0]);
                    i++;
                }
            }


            observationMatrixList.Clear();
            for (int observationIndex =
                    0;
                observationIndex < brainParameters.cameraResolutions.Count();
                observationIndex++)
            {
                texturesHolder.Clear();
                foreach (Agent agent in agentList)
                {
                    texturesHolder.Add(agentInfo[agent].visualObservations[observationIndex]);
                }

                observationMatrixList.Add(
                    BatchVisualObservations(texturesHolder,
                        brainParameters.cameraResolutions[observationIndex].blackAndWhite));
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
                throw new UnityAgentsException(string.Format(
                    @"The node {0} could not be found. Please make sure the graphScope {1} is correct",
                    graphScope + ActionPlaceholderName, graphScope));
            }

            if (hasBatchSize)
            {
                runner.AddInput(graph[graphScope + BatchSizePlaceholderName][0],
                    new int[] {currentBatchSize});
            }

//        foreach (TensorFlowAgentPlaceholder placeholder in graphPlaceholders)
//        {
//            try
//            {
//                if (placeholder.valueType == TensorFlowAgentPlaceholder.tensorType.FloatingPoint)
//                {
//                    runner.AddInput(graph[graphScope + placeholder.name][0], new float[] { Random.Range(placeholder.minValue, placeholder.maxValue) });
//                }
//                else if (placeholder.valueType == TensorFlowAgentPlaceholder.tensorType.Integer)
//                {
//                    runner.AddInput(graph[graphScope + placeholder.name][0], new int[] { Random.Range((int)placeholder.minValue, (int)placeholder.maxValue + 1) });
//                }
//            }
//            catch
//            {
//                throw new UnityAgentsException(string.Format(@"One of the Tensorflow placeholder cound nout be found.
//                In brain {0}, there are no {1} placeholder named {2}.",
//                        brain.gameObject.name, placeholder.valueType.ToString(), graphScope + placeholder.name));
//            }
//        }

            // Create the state tensor
            if (hasState)
            {
                if (brainParameters.vectorObservationSpaceType == SpaceType.discrete)
                {
                    var discreteInputState = new int[currentBatchSize, 1];
                    for (int i = 0; i < currentBatchSize; i++)
                    {
                        discreteInputState[i, 0] = (int) inputState[i, 0];
                    }

                    runner.AddInput(graph[graphScope + VectorObservationPlacholderName][0],
                        discreteInputState);
                }
                else
                {
                    runner.AddInput(graph[graphScope + VectorObservationPlacholderName][0],
                        inputState);
                }
            }

            // Create the previous action tensor
            if (hasPrevAction)
            {
                runner.AddInput(graph[graphScope + PreviousActionPlaceholderName][0],
                    inputPrevAction);
            }

            // Create the observation tensors
            for (int obs_number =
                    0;
                obs_number < brainParameters.cameraResolutions.Length;
                obs_number++)
            {
                runner.AddInput(graph[graphScope + VisualObservationPlaceholderName[obs_number]][0],
                    observationMatrixList[obs_number]);
            }

            if (hasRecurrent)
            {
                runner.AddInput(graph[graphScope + "sequence_length"][0], 1);
                runner.AddInput(graph[graphScope + RecurrentInPlaceholderName][0],
                    inputOldMemories);
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
                    errorMessage =
                        string.Format(@"The tensorflow graph needs an input for {0} of type {1}",
                            e.Message.Split(new string[] {"Node: "}, 0)[1].Split('=')[0],
                            e.Message.Split(new string[] {"dtype="}, 0)[1].Split(',')[0]);
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

            if (brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                var output = networkOutput[0].GetValue() as float[,];
                var i = 0;
                foreach (Agent agent in agentList)
                {
                    var a = new float[brainParameters.vectorActionSize];
                    for (int j = 0; j < brainParameters.vectorActionSize; j++)
                    {
                        a[j] = output[i, j];
                    }

                    agent.UpdateVectorAction(a);
                    i++;
                }
            }
            else if (brainParameters.vectorActionSpaceType == SpaceType.discrete)
            {
                long[,] output = networkOutput[0].GetValue() as long[,];
                var i = 0;
                foreach (Agent agent in agentList)
                {
                    var a = new float[1] {(float) (output[i, 0])};
                    agent.UpdateVectorAction(a);
                    i++;
                }
            }


#else
            if (agentInfo.Count > 0)
            {
                throw new UnityAgentsException(string.Format(
                    @"The brain {0} was set to Internal but the Tensorflow 
                        library is not present in the Unity project.",
                    brain.gameObject.name));
            }
#endif
            agentInfo.Clear();
        }

        /// <summary>
        /// Converts a list of Texture2D into a Tensor.
        /// </summary>
        /// <returns>
        /// A 4 dimensional float Tensor of dimension
        /// [batch_size, height, width, channel].
        /// Where batch_size is the number of input textures,
        /// height corresponds to the height of the texture,
        /// width corresponds to the width of the texture,
        /// channel corresponds to the number of channels extracted from the
        /// input textures (based on the input blackAndWhite flag
        /// (3 if the flag is false, 1 otherwise).
        /// The values of the Tensor are between 0 and 1.
        /// </returns>
        /// <param name="textures">
        /// The list of textures to be put into the tensor.
        /// Note that the textures must have same width and height.
        /// </param>
        /// <param name="blackAndWhite">
        /// If set to <c>true</c> the textures
        /// will be converted to grayscale before being stored in the tensor.
        /// </param>
        public static float[,,,] BatchVisualObservations(
            List<Texture2D> textures, bool blackAndWhite)
        {
            int batchSize = textures.Count();
            int width = textures[0].width;
            int height = textures[0].height;
            int pixels = 0;
            if (blackAndWhite)
                pixels = 1;
            else
                pixels = 3;
            float[,,,] result = new float[batchSize, height, width, pixels];

            for (int b = 0; b < batchSize; b++)
            {
                Color32[] cc = textures[b].GetPixels32();
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        Color32 currentPixel = cc[h * width + w];
                        if (!blackAndWhite)
                        {
                            // For Color32, the r, g and b values are between
                            // 0 and 255.
                            result[b, textures[b].height - h - 1, w, 0] =
                                currentPixel.r / 255.0f;
                            result[b, textures[b].height - h - 1, w, 1] =
                                currentPixel.g / 255.0f;
                            result[b, textures[b].height - h - 1, w, 2] =
                                currentPixel.b / 255.0f;
                        }
                        else
                        {
                            result[b, textures[b].height - h - 1, w, 0] =
                                (currentPixel.r + currentPixel.g + currentPixel.b) / 3.0f;
                        }
                    }
                }
            }

            return result;
        }
    }
}