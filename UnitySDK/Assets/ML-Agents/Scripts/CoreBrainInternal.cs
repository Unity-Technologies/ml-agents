using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Security.Principal;
using NUnit.Framework;
using UnityEngine.MachineLearning.InferenceEngine;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using UnityEngine.Rendering;
// TODO : Remove
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace MLAgents
{
    /// CoreBrain which decides actions using internally embedded TensorFlow model.
    public class CoreBrainInternal : ScriptableObject, CoreBrain
    {
        [SerializeField] [Tooltip("If checked, the brain will broadcast states and actions to Python.")]
#pragma warning disable
        private bool broadcast = true;
#pragma warning restore


        Batcher brainBatcher;


        const string BatchSizePlaceholderName = "batch_size";
        const string SequenceLengthPlaceholderName = "sequence_length";
        const string VectorObservationPlacholderName = "vector_observation";
        const string RecurrentInPlaceholderName = "recurrent_in";
        const string VisualObservationPlaceholderNamePrefix = "visual_observation_";
        const string PreviousActionPlaceholderName = "prev_action";
        const string ActionMaskPlaceholderName = "action_masks";
        const string RandomNormalEpsilonPlaceholderName = "random_normal_epsilon";

        const string ValueEstimateOutputName = "value_estimate";
        const string RecurrentOutOutputName = "recurrent_out";
        const string MemorySizeName = "memory_size";
        const string kApiVersion = "api_version";
        const string ActionOutputName = "action";

        private long _memorySize = 0;
        private int _actionSize = 0;
        private RandomNormal _randomNormal = new RandomNormal(1);

        public Model m_model;

        InferenceEngine m_engine;
        private Tensor[] inputs;
        private Tensor[] outputs;

        public Brain brain;

        /// Create the reference to the brain
        public void SetBrain(Brain b)
        {
            brain = b;
        }

        /// Loads the tensorflow graph model to generate a TFGraph object
        public void InitializeCoreBrain(MLAgents.Batcher brainBatcher)
        {

#if UNITY_ANDROID && !UNITY_EDITOR
// This needs to ba called only once and will raise an exception if 
// there are multiple internal brains
        try{
            TensorFlowSharp.Android.NativeBinding.Init();
        }
        catch{
            
        }
#endif
            if ((brainBatcher == null)
                || (!broadcast))
            {
                this.brainBatcher = null;
            }
            else
            {
                this.brainBatcher = brainBatcher;
                this.brainBatcher.SubscribeBrain(brain.gameObject.name);
            }

            if (m_model != null)
            {
                
                InferenceEngineConfig config;
                m_engine = InferenceAPI.LoadModel(m_model, config);

                Debug.Log(m_engine);
                // Generate the Input tensors
                

                inputs = /*m_engine.InputFeatures();*/ TMP_GetInputTensors();

                var inputShapeError = TestInputTensorShape(
                    inputs, 
                    brain.brainParameters);
                if (inputShapeError.Count > 0)
                {
                    throw new UnityAgentsException(inputShapeError[0]);
                }
                
                // TODO : Generate the visual_observations prefix
                // TODO : Put all the shape tests here
                // TODO : Generate output Tensors
                // TODO : Get some of the outputs out like the API #
               
                
                // Generate the Output tensors
                var AllOutputs =/*m_engine.OutputFeatures();*/ TMP_GetOutputTensors();
                List<Tensor> inferenceOutputs = new List<Tensor>();
                foreach (var tensor in AllOutputs)
                {
                    if ((tensor.Name == ValueEstimateOutputName) ||
                        (tensor.Name == RecurrentOutOutputName) ||
                        (tensor.Name == ActionOutputName))
                    {
                        inferenceOutputs.Add(tensor);
                    }
                }
                outputs = inferenceOutputs.ToArray();
                // TODO : Compare with real value;

                if ( /*memory_size is part of the outputs*/ false)
                {
                    Tensor memoryTensor = new Tensor
                    {
                        Name = MemorySizeName,
                        ValueType = Tensor.TensorType.Integer,
                        Shape = new long[1] {1}
                    };
                    m_engine.ExecuteGraph(new Tensor[0], new Tensor[1] {memoryTensor});
                    _memorySize = (memoryTensor.Data as long[])[0];
                    Assert.IsNotNull(_memorySize);
                }


                if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
                {
                    _actionSize = brain.brainParameters.vectorActionSize[0];
                }
                else
                {
                    _actionSize = brain.brainParameters.vectorActionSize.Length;
                }
            }
            else
            {
                // TODO : Implement
                throw new UnityAgentsException("ERROR TO IMPLEMENT");
            }
        }

        private Tensor[] TMP_GetInputTensors()
        {
            return new Tensor[2]
            {
                new Tensor()
                  {
                      Name = VectorObservationPlacholderName,
                      Shape = new long[2]
                      {
                          12, 8
                      },
                      ValueType = Tensor.TensorType.FloatingPoint
                  },
                new Tensor()
                {
                    Name = RandomNormalEpsilonPlaceholderName,
                    Shape = new long[2]
                    {
                        12, 2
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[12, 2]
                },
            };
        }
        
        private Tensor[] TMP_GetOutputTensors()
        {
            return new Tensor[1]
            {
                new Tensor()
                {
                    Name = ActionOutputName,
                    Shape = new long[2]
                    {
                        12, brain.brainParameters.vectorActionSize[0]
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[12, brain.brainParameters.vectorActionSize[0]]
                }
            };
        }

        // Put this into a separate class
        private static List<string> TestInputTensorShape(
            Tensor[] tensors, 
            BrainParameters brainParams)
        {
            List<string> result = new List<string>();
            foreach (var tensor in tensors)
            {
                switch (tensor.Name)
                {
                    case VectorObservationPlacholderName:
                        result.Add(TestVectorObsShape(tensor, brainParams));
                        break;
                    case PreviousActionPlaceholderName:
                        // TODO : Discrete case
                        result.Add(TestPreviousActionShape(tensor, brainParams));
                        break;
                }
            }
            return result.Where(x => x!=null).ToList();
        }

        private static string TestVectorObsShape(
            Tensor tensor,
            BrainParameters brainParams)
        {
            var vecObsSizeBp = brainParams.vectorObservationSize;
            var numStackedVector = brainParams.numStackedVectorObservations;
            var totalVecObsSizeT = tensor.Shape[1];
            if (vecObsSizeBp * numStackedVector != totalVecObsSizeT)
            {
                return string.Format(
                    "Vector Observation Size of the model does not match. " +
                    "Received {0} x {1} but was expecting {2}.",
                    vecObsSizeBp, numStackedVector, totalVecObsSizeT);
            }
            return null;
        }
        
        private static string TestPreviousActionShape(
            Tensor tensor,
            BrainParameters brainParams)
        {
            var numberActionsBp = brainParams.vectorActionSize.Length;
            var numberActionsT = tensor.Shape[1];
            if  (numberActionsBp != numberActionsT)
            {
                return string.Format(
                    "Action Size of the model does not match. " +
                    "Received {0} but was expecting {2}.",
                    numberActionsBp, numberActionsT);
            }
            return null;
        }

        /// Uses the stored information to run the tensorflow graph and generate 
        /// the actions.
        public void DecideAction(Dictionary<Agent, AgentInfo> agentInfo)
        {
            if (brainBatcher != null)
            {
                brainBatcher.SendBrainInfo(brain.gameObject.name, agentInfo);
            }

            var currentBatchSize = agentInfo.Count();
            var agentList = agentInfo.Keys.ToList();
            if (currentBatchSize == 0)
            {
                return;
            }

            // Generating the Input tensors
            foreach (var tensor in inputs)
            {
                switch (tensor.Name)
                {
                    case BatchSizePlaceholderName:
                        // TODO : Put all these into helper methods ?
                        tensor.Data = new long[1] {currentBatchSize};
                        break;
                    case SequenceLengthPlaceholderName:
                        tensor.Data = new long[1] {1};
                        break;
                    case VectorObservationPlacholderName:
                        var vecObsSizeT = tensor.Shape[1];
                        tensor.Data = new float[currentBatchSize, vecObsSizeT];
                        for (var agentIndex = 0 ; agentIndex < agentList.Count; agentIndex++)
                        {
                            var vectorObs = 
                                agentInfo[agentList[agentIndex]].stackedVectorObservation;
                            for (var j = 0; j < vecObsSizeT; j++)
                            {
                                tensor.Data.SetValue(vectorObs[j], new int[2]{agentIndex, j});
                            }
                        }
                        break;
                    case RecurrentInPlaceholderName:
                        tensor.Data = new float[currentBatchSize, _memorySize];
                        for (var agentIndex = 0 ; agentIndex< agentList.Count; agentIndex++)
                        {
                            var m = agentInfo[agentList[agentIndex]].memories.ToArray();
                            for (var j = 0; j < m.Length; j++)
                            {
                                tensor.Data.SetValue(m[j], new int[2]{agentIndex, j});
                            }
                        }
                        break;
                    case PreviousActionPlaceholderName:
                        var numberActionsBp = brain.brainParameters.vectorActionSize.Length;
                        tensor.Data = new int[currentBatchSize, numberActionsBp];
                        for (var agentIndex = 0 ; agentIndex< agentList.Count; agentIndex++)
                        {
                            var actionList = agentInfo[agentList[agentIndex]].storedVectorActions;
                            for (var j = 0 ; j < numberActionsBp; j++)
                            {
                                tensor.Data.SetValue(actionList[j], new int[2]{agentIndex, j});
                            }
                        }
                        break;
                    case ActionMaskPlaceholderName:
                        var totalNumberActionsBp = brain.brainParameters.vectorActionSize.Sum();
                        tensor.Data = new float[currentBatchSize, totalNumberActionsBp];
                        for (var agentIndex = 0 ; agentIndex< agentList.Count; agentIndex++)
                        {
                            for (var j = 0; j < totalNumberActionsBp; j++)
                            {
                                var maskList = agentInfo[agentList[agentIndex]].actionMasks;
                                if (maskList != null)
                                {
                                    tensor.Data.SetValue(
                                        maskList[j] ? 0.0f : 1.0f, 
                                        new int[2]{agentIndex, j});
                                }
                                else
                                {
                                    tensor.Data.SetValue(
                                        1.0f, 
                                        new int[2]{agentIndex, j});
                                }
                            }
                        }
                        break;
                    case RandomNormalEpsilonPlaceholderName:
                        _randomNormal.FillTensor(tensor);
                        break;
                    //TODO : The Visual Observations 
                    
                    default:
                        // TODO : Implement
                        throw new UnityAgentsException("Error to implement");
                }
            }
            
            // Execute the Model
            m_engine.ExecuteGraph(inputs, outputs);

            foreach (var tensor in outputs)
            {
                switch (tensor.Name)
                {
                    case ActionOutputName:
                        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
                        {
                            var tensorDataAction = tensor.Data as float[,];
                            for (var agentIndex = 0; agentIndex < agentList.Count; agentIndex++)
                            {
                                var a = new float[_actionSize];
                                for (var j = 0; j < _actionSize; j++)
                                {
                                    a[j] = tensorDataAction[agentIndex, j];
                                }
                                agentList[agentIndex].UpdateVectorAction(a);
                            }
                        }
                        else
                        {
                            // TODO : Implement
                            throw new UnityAgentsException("Error to Implement");
                        }
                        break;
                    case RecurrentOutOutputName:
                        var tensorDataMemory = tensor.Data as float[,];
                        for (var agentIndex = 0 ; agentIndex< agentList.Count; agentIndex++)
                        {
                            var a = new float[_memorySize];
                            for (var j = 0; j < _memorySize; j++)
                            {
                                a[j] = tensorDataMemory[agentIndex, j];
                            }
                            // TODO Make better
                            agentList[agentIndex].UpdateMemoriesAction(a.ToList());
                        }
                        break;
                    case ValueEstimateOutputName:
                        var tensorDataValue = tensor.Data as float[,];
                        for (var agentIndex = 0; agentIndex < agentList.Count; agentIndex++)
                        {
                            agentList[agentIndex].UpdateValueAction(tensorDataValue[agentIndex, 0]);
                        }
                        break;
                    default:
                        // TODO : Implement
                        throw new UnityAgentsException("Error to implement");
                }
            }
        }

        /// Displays the parameters of the CoreBrainInternal in the Inspector 
        public void OnInspector()
        {
#if UNITY_EDITOR
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            broadcast = EditorGUILayout.Toggle(new GUIContent("Broadcast",
                "If checked, the brain will broadcast states and actions to Python."), broadcast);

            var serializedBrain = new SerializedObject(this);
            GUILayout.Label("Edit the Tensorflow graph parameters here");
            var tfGraphModel = serializedBrain.FindProperty("m_model");
            serializedBrain.Update();
            EditorGUILayout.ObjectField(tfGraphModel);
            serializedBrain.ApplyModifiedProperties();
            var errors = TestInputTensorShape(
                TMP_GetInputTensors(),
                brain.brainParameters);
            foreach (var error in errors)
            {
                if (error != null)
                    EditorGUILayout.HelpBox(error, MessageType.Error);
            }
#endif
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
            float[] resultTemp = new float[batchSize * height * width * pixels];
            int hwp = height * width * pixels;
            int wp = width * pixels;

            for (int b = 0; b < batchSize; b++)
            {
                Color32[] cc = textures[b].GetPixels32();
                for (int h = height - 1; h >= 0; h--)
                {
                    for (int w = 0; w < width; w++)
                    {
                        Color32 currentPixel = cc[(height - h - 1) * width + w];
                        if (!blackAndWhite)
                        {
                            // For Color32, the r, g and b values are between
                            // 0 and 255.
                            resultTemp[b * hwp + h * wp + w * pixels] = currentPixel.r / 255.0f;
                            resultTemp[b * hwp + h * wp + w * pixels + 1] = currentPixel.g / 255.0f;
                            resultTemp[b * hwp + h * wp + w * pixels + 2] = currentPixel.b / 255.0f;
                        }
                        else
                        {
                            resultTemp[b * hwp + h * wp + w * pixels] =
                                (currentPixel.r + currentPixel.g + currentPixel.b)
                                / 3f / 255.0f;
                        }
                    }
                }
            }

            System.Buffer.BlockCopy(resultTemp, 0, result, 0, batchSize * hwp * sizeof(float));
            return result;
        }
    }
}