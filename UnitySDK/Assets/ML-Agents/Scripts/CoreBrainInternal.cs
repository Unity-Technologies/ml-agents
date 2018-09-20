using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Text;
using NUnit.Framework;
using UnityEngine.MachineLearning.InferenceEngine;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using UnityEngine.Playables;
// TODO : Remove
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace MLAgents
{
    /// CoreBrain which decides actions using internally embedded TensorFlow model.
    public class CoreBrainInternal : ScriptableObject, CoreBrain
    {
        private const long version = 1; 
        
        private class NodeNames
        {
            public string BatchSizePlaceholder = "batch_size";
            public string SequenceLengthPlaceholder = "sequence_length";
            public string VectorObservationPlacholder = "vector_observation";
            public string RecurrentInPlaceholder = "recurrent_in";
            public string VisualObservationPlaceholderPrefix = "visual_observation_";
            public string PreviousActionPlaceholder = "prev_action";
            public string ActionMaskPlaceholder = "action_masks";
            public string RandomNormalEpsilonPlaceholder = "epsilon";

            public string ValueEstimateOutput = "value_estimate";
            public string RecurrentOutOutput = "recurrent_out";
            public string MemorySize = "memory_size";
            public string VersionNumber = "version_number";
            public string IsContinuousControl = "is_continuous_control";
            public string ActionOutput = "action";
        }
        
        [SerializeField] [Tooltip("If checked, the brain will broadcast states and actions to Python.")]
#pragma warning disable
        private bool broadcast = true;
#pragma warning restore


        private List<string> tmp_editor_errors;
        
        Batcher brainBatcher;

        NodeNames _nodeNames = new NodeNames();
        

        private long _memorySize;
        private long _modelVersionNumber;
        private long _isContinuous;

        private Dictionary<string, Action<Tensor, int, Dictionary<Agent, AgentInfo>>>
            _inputTensorGenerators;

        private Dictionary<string, Action<Tensor, Dictionary<Agent, AgentInfo>>>
            _outputTensorAppliers;
        
        
        private RandomNormal _randomNormal = new RandomNormal(0);
        private Multinomial _multinomial = new Multinomial(0);

        public Model m_model;

        InferenceEngine m_engine;
        private Tensor[] inferenceInputs;
        private Tensor[] inferenceOutputs;
//        private Tensor[] allOutputs;

        public Brain brain;

        /// Create the reference to the brain
        public void SetBrain(Brain b)
        {
            brain = b;
        }

        /// Loads the tensorflow graph model to generate a TFGraph object
        public void InitializeCoreBrain(Batcher brainBatcher)
        {
            
            Debug.Log(TensorFlow.TFCore.Version);

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

            InitializeModel(m_model);
            UpdateInputGenerator();
            UpdateOutputAppliers();

        }

        private void UpdateInputGenerator()
        {
            _inputTensorGenerators = new Dictionary<string, 
                Action<Tensor, int, Dictionary<Agent, AgentInfo>>>();
            _inputTensorGenerators[_nodeNames.BatchSizePlaceholder] = GenerateBatchSize;
            _inputTensorGenerators[_nodeNames.SequenceLengthPlaceholder] = GenerateSequenceLength;
            _inputTensorGenerators[_nodeNames.VectorObservationPlacholder] = GenerateVectorObservation;
            _inputTensorGenerators[_nodeNames.RecurrentInPlaceholder] = GenerateRecurrentInput;
            _inputTensorGenerators[_nodeNames.PreviousActionPlaceholder] = GeneratePreviousActionInput;
            _inputTensorGenerators[_nodeNames.ActionMaskPlaceholder] = GenerateActionMaskInput;
            _inputTensorGenerators[_nodeNames.RandomNormalEpsilonPlaceholder] =
                (tensor, batchSize, agentInfo) =>
                    GenerateRandomNormalInput(tensor, batchSize, agentInfo, _randomNormal);
            for (var visIndex = 0;
                visIndex < brain.brainParameters.cameraResolutions.Length;
                visIndex++)
            {
                var index = visIndex;
                var bw = brain.brainParameters.cameraResolutions[visIndex].blackAndWhite;
                _inputTensorGenerators[_nodeNames.VisualObservationPlaceholderPrefix + visIndex] =
                    (tensor, batchSize, agentInfo) =>
                        GenerateVisualObservationInput(tensor, agentInfo, index, bw);
            }
        }
        
        private void UpdateOutputAppliers()
        {
            _outputTensorAppliers = new Dictionary<string, 
                Action<Tensor, Dictionary<Agent, AgentInfo>>>();
            
            _outputTensorAppliers[_nodeNames.ValueEstimateOutput] = ApplyValueEstimate;
            if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                _outputTensorAppliers[_nodeNames.ActionOutput] = ApplyContinuousActionOutput;
            }
            else
            {
                _outputTensorAppliers[_nodeNames.ActionOutput] = (tensor, agentInfo) =>
                    ApplyDiscreteActionOutput(tensor, agentInfo, _multinomial,
                        brain.brainParameters.vectorActionSize);
            }
            _outputTensorAppliers[_nodeNames.RecurrentOutOutput] = ApplyMemoryOutput;
            
            
        }

        private void InitializeModel(Model model)
        {
            if (model != null)
            {

                Debug.Log("Initialize");
                InferenceEngineConfig config;
                m_engine = InferenceAPI.LoadModel(model, config);

                // Generate the Input tensors
                inferenceInputs = GetInputTensors();
                inferenceOutputs = GetOutputTensors().ToArray();

//                _modelVersionNumber = GetModelData(m_engine, _nodeNames.VersionNumber);
                _memorySize = GetModelData(m_engine, _nodeNames.MemorySize);
//                _isContinuous = GetModelData(m_engine, _nodeNames.IsContinuousControl);
            }
            else
            {
                m_engine = null;
                // TODO : Implement
                throw new UnityAgentsException("ERROR TO IMPLEMENT : There was no model");
            }
        }

        private static long GetModelData(InferenceEngine engine, string name)
        {
            try
            {
                var outputs = new Tensor[]
                {
                    new Tensor()
                    {
                        Name = name,
                        ValueType = Tensor.TensorType.Integer,
                        Shape = new long[]{},
                        Data = new long[1]
                    },
                };
                engine.ExecuteGraph(new Tensor[0], outputs);
           
                return (outputs[0].Data as int[])[0];
            }
            catch
            {
                Debug.LogError("Node not in graph " + name);
                return -1;
            }
        }

        public List<string> GetModelErrors()
        {
            if (m_engine == null)
            {
                return new List<string>();
            }
            
            // TODO : When the method is called, the engine and the brainParameters must be up to date
            inferenceInputs = GetInputTensors();
            inferenceOutputs = GetOutputTensors().ToArray();

            var errors = new List<string>();
            if (_modelVersionNumber != version)
            {
                // TODO : Make better
                errors.Add("Model has not been trained using the same version as the " +
                           "Internal Brain");
            }

            if (_isContinuous == 1 &&
                brain.brainParameters.vectorActionSpaceType != SpaceType.continuous)
            {
                // TODO : Implement
//                errors.Add("Error to implement");
            }
            errors.AddRange(TestInputTensorShape(
                inferenceInputs,
                brain.brainParameters,
                _nodeNames));
            errors.AddRange(TestInputTensorPresence(
                inferenceInputs,
                brain.brainParameters,
                _nodeNames));
            errors.AddRange(TestOutputTensorPresence(
                inferenceOutputs,
                brain.brainParameters,
                _nodeNames));
            return errors;
        }

        private Tensor[] GetInputTensors()
        {
            var n_agents = 2;
            BrainParameters bp = brain.brainParameters;

            var tensorList = new List<Tensor>();
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorList.Add(
                    new Tensor()
                {
                    Name = _nodeNames.RandomNormalEpsilonPlaceholder,
                    Shape = new long[]
                    {
                        n_agents, bp.vectorActionSize[0]
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[n_agents, bp.vectorActionSize[0]]
                });
            }
            else
            {
                tensorList.Add(
                    new Tensor()
                {
                    Name = _nodeNames.ActionMaskPlaceholder,
                    Shape = new long[]{n_agents, bp.vectorActionSize.Sum()},
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[n_agents, bp.vectorActionSize.Sum()]
                });
            }
            foreach(var res in bp.cameraResolutions)
            {
                tensorList.Add(new Tensor()
                {
                    Name = _nodeNames.VisualObservationPlaceholderPrefix + "0",
                    Shape = new long[4]
                    {
                        n_agents, res.width, res.height, res.blackAndWhite ? 1 : 3
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[n_agents, res.width, res.height, res.blackAndWhite ? 1 : 3]
                });
            }

            if (bp.vectorObservationSize > 0)
            {
                tensorList.Add(new Tensor()
                  {
                      Name = _nodeNames.VectorObservationPlacholder,
                      Shape = new long[2]
                      {
                          n_agents, bp.vectorObservationSize * bp.numStackedVectorObservations
                      },
                      ValueType = Tensor.TensorType.FloatingPoint
                  });
            }

            if (_memorySize > 0)
            {
                tensorList.Add(new Tensor()
                {
                    Name = _nodeNames.RecurrentInPlaceholder,
                    Shape = new long[2]
                    {
                        n_agents, _memorySize
                    },
                    ValueType = Tensor.TensorType.FloatingPoint
                });
                tensorList.Add(new Tensor()
                {
                    Name = _nodeNames.SequenceLengthPlaceholder,
                    Shape = new long[]
                    {
                        
                    },
                    ValueType = Tensor.TensorType.Integer
                });
                if (brain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
                {
                    tensorList.Add(new Tensor()
                    {
                        Name = _nodeNames.PreviousActionPlaceholder,
                        Shape = new long[2]
                        {
                            n_agents, brain.brainParameters.vectorActionSize.Length
                        },
                        ValueType = Tensor.TensorType.Integer
                    });
                }
            }

            return tensorList.ToArray();
        }
        
        private List<Tensor> GetOutputTensors()
        {
            var n_agents = 16;
            BrainParameters bp = brain.brainParameters;
            var tensorList = new List<Tensor>();
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorList.Add(new Tensor()
                {
                    Name = _nodeNames.ActionOutput,
                    Shape = new long[]
                    {
                        n_agents, bp.vectorActionSize[0]
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[n_agents, bp.vectorActionSize[0]]
                });
            }
            else
            {
                tensorList.Add(
                    new Tensor()
                    {
                        Name = _nodeNames.ActionOutput,
                        Shape = new long[]{n_agents, bp.vectorActionSize.Sum()},
                        ValueType = Tensor.TensorType.FloatingPoint,
                        Data = new float[n_agents, bp.vectorActionSize.Sum()]
                    });
            }
            if (_memorySize > 0)
            {
                tensorList.Add(new Tensor()
                {
                    Name = _nodeNames.RecurrentOutOutput,
                    Shape = new long[2]
                    {
                        n_agents, _memorySize
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[n_agents, _memorySize]
                });
            }

            return tensorList;

        }

        private static List<string> TestInputTensorPresence(
            IEnumerable<Tensor> tensors,
            BrainParameters brainParams,
            NodeNames nodeNames)
        {
            var result = new List<string>();
            var tensorsNames = tensors.Select(x => x.Name);

            // If there is no Vector Observation Input but the Brain Parameters expect one.
            if ((brainParams.vectorObservationSize != 0) &&
                (!tensorsNames.Contains(nodeNames.VectorObservationPlacholder)))
            {
                result.Add("The model does not contain a Vector Observation Placeholder Input. " +
                           "You must set the Vector Observation Space Size to 0.");
            }

            for (var visObsIndex = 0;
                visObsIndex < brainParams.cameraResolutions.Length;
                visObsIndex++)
            {
                if (!tensorsNames.Contains(
                    nodeNames.VisualObservationPlaceholderPrefix + visObsIndex))
                {
                    result.Add("The model does not contain a Visual Observation Placeholder" +
                               " Input for visual observation "+visObsIndex+".");
                }
            }
            return result;
        }
        
        private static List<string> TestOutputTensorPresence(
            IEnumerable<Tensor> tensors,
            BrainParameters brainParams,
            NodeNames nodeNames)
        {
            var result = new List<string>();
            var tensorsNames = tensors.Select(x => x.Name);

            // If there is no Action Output.
            if (!tensorsNames.Contains(nodeNames.ActionOutput))
            {
                result.Add("The model does not contain an Action Output Node.");
            }

            return result;
        }

        private static List<string> TestInputTensorShape(
            IEnumerable<Tensor> tensors, 
            BrainParameters brainParams,
            NodeNames nodeNames)
        {
            var result = new List<string>();
 
            var tensorTester =
                new Dictionary<string, Func<Tensor, BrainParameters, string>>()
                {
                    {nodeNames.VectorObservationPlacholder, TestVectorObsShape},
                    {nodeNames.PreviousActionPlaceholder, TestPreviousActionShape},
                    {nodeNames.RandomNormalEpsilonPlaceholder, ((tensor, parameters) => null)},
                    // TODO : Need a tester for some of those
                    {nodeNames.ActionMaskPlaceholder, ((tensor, parameters) => null)},
                    {nodeNames.SequenceLengthPlaceholder, ((tensor, parameters) => null)},
                    {nodeNames.RecurrentInPlaceholder, ((tensor, parameters) => null)},
                };

            for (var visObsIndex = 0;
                visObsIndex < brainParams.cameraResolutions.Length; 
                visObsIndex++)
            {
                var index = visObsIndex;
                tensorTester[nodeNames.VisualObservationPlaceholderPrefix + visObsIndex] =
                    (tensor, bp) => TestVisualObsShape(tensor, bp, index);
            }
            
            foreach (var tensor in tensors)
            {
                if (!tensorTester.ContainsKey(tensor.Name))
                {
                    result.Add("No placeholder for input : " + tensor.Name);
                }
                else
                {
                    var tester = tensorTester[tensor.Name];
                    var error = tester.Invoke(tensor, brainParams);
                    if (error != null)
                    {
                        result.Add(error);
                    }
                }
            }
            return result;
        }
        
        private static string TestActionMaskShape(
            Tensor tensor,
            BrainParameters brainParams)
        {
            Debug.Log("error to implement.");
            return null;
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
        
        private static string TestVisualObsShape(
            Tensor tensor,
            BrainParameters brainParams,
            int visObsIndex)
        {
            
            var resolutionBp = brainParams.cameraResolutions[visObsIndex];
            var widthBp = resolutionBp.width;
            var heightBp = resolutionBp.height;
            var pixelBp = resolutionBp.blackAndWhite ? 1 : 3;
            var widthT = tensor.Shape[1];
            var heightT = tensor.Shape[2];
            var pixelT = tensor.Shape[3];
            if  ((widthBp != widthT) || (heightBp != heightT) || (pixelBp != pixelT))
            {
                return string.Format(
                    "The visual Observation {0} of the model does not match. " +
                    "Received Tensor of shape [?x{1}x{2}x{3}] but was expecting [?x{4}x{5}x{6}].",
                    visObsIndex, widthBp, heightBp, pixelBp, widthT, heightT, pixelT);
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
            if (currentBatchSize == 0)
            {
                return;
            }

            // Generating the Input tensors
            for (var tensorIndex = 0; tensorIndex<inferenceInputs.Length; tensorIndex++)
            {
                var tensor = inferenceInputs[tensorIndex];
                if (!_inputTensorGenerators.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException("Error to implement.");
                }
                _inputTensorGenerators[tensor.Name].Invoke(tensor, currentBatchSize, agentInfo);

            }
            

            // Execute the Model
            m_engine.ExecuteGraph(inferenceInputs, inferenceOutputs);


            // Update the outputs
            for (var tensorIndex = 0; tensorIndex<inferenceOutputs.Length; tensorIndex++)
            {
                var tensor = inferenceOutputs[tensorIndex];
                if (!_outputTensorAppliers.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException("Error to implement.");
                }
                _outputTensorAppliers[tensor.Name].Invoke(tensor, agentInfo);
                
            }
            Debug.Log("Execute Graph "+Time.timeScale);
        }
        
        private static void GenerateBatchSize(
                Tensor tensor,
                int batchSize,
                Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Data = new int[] {batchSize};
        }
        
        private static void GenerateSequenceLength(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Data = new int[] {1};
        }
            
        private static void GenerateVectorObservation(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var vecObsSizeT = tensor.Shape[1];
            tensor.Data = new float[batchSize, vecObsSizeT];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var vectorObs = agentInfo[agent].stackedVectorObservation;
                for (var j = 0; j < vecObsSizeT; j++)
                {
                    tensor.Data.SetValue(vectorObs[j], new int[2] {agentIndex, j});
                }
                agentIndex++;
            }
        }
        
        private static void GenerateRecurrentInput(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var memorySize = tensor.Shape[1];
            tensor.Data = new float[batchSize, memorySize];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var memory = agentInfo[agent].memories;
                for (var j = 0; j < memorySize; j++)
                {
                    if (j >= memory.Count)
                    {
                        break;
                    }
                    tensor.Data.SetValue(memory[j], new int[2] {agentIndex, j});
                }
                agentIndex++;
            }
        }
        
        private static void GeneratePreviousActionInput(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var actionSize = tensor.Shape[1];
            tensor.Data = new int[batchSize, actionSize];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var pastAction = agentInfo[agent].storedVectorActions;
                for (var j = 0; j < actionSize; j++)
                {
                    tensor.Data.SetValue((int)pastAction[j], new int[2] {agentIndex, j});
                }
                agentIndex++;
            }
        }
        
        private static void GenerateActionMaskInput(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var maskSize = tensor.Shape[1];
            tensor.Data = new float[batchSize, maskSize];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var maskList = agentInfo[agent].actionMasks;
                for (var j = 0; j < maskSize; j++)
                {
                    if (maskList != null)
                    {
                        tensor.Data.SetValue(
                            maskList[j] ? 0.0f : 1.0f,
                            new int[2] {agentIndex, j});
                    }
                    else
                    {
                        tensor.Data.SetValue(
                            1.0f,
                            new int[2] {agentIndex, j});
                    }
                }
                agentIndex++;
            }
        }

        private static void GenerateRandomNormalInput(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo,
            RandomNormal randomNormal)
        {
            tensor.Shape[0] = batchSize;
            var actionSize = tensor.Shape[1];
            tensor.Data = new float[batchSize, actionSize];
            randomNormal.FillTensor(tensor);
        }

        private static void GenerateVisualObservationInput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo,
            int visIndex,
            bool bw)
        {
            //TODO : More efficient ?
            var textures = agentInfo.Keys.Select(
                agent => agentInfo[agent].visualObservations[visIndex]).ToList();
            tensor.Data = BatchVisualObservations(textures, bw);

        }

        private static void ApplyContinuousActionOutput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataAction = tensor.Data as float[,];
            var actionSize = tensor.Shape[1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var a = new float[actionSize];
                for (var j = 0; j < actionSize; j++)
                {
                    a[j] = tensorDataAction[agentIndex, j];
                }

                agent.UpdateVectorAction(a);
                agentIndex++;
            }
        }

        /// <summary>
        /// Generates an array containing the starting indicies of each branch in the vector action
        /// Makes a cumulative sum.
        /// </summary>
        /// <returns></returns>
        private static int[] CreateActionStartinIndices(int[] vectorActionSize)
        {
            var runningSum = 0;
            var result = new int[vectorActionSize.Length + 1];
            for (var actionIndex = 0;
                actionIndex < vectorActionSize.Length; actionIndex++)
            {
                runningSum += vectorActionSize[actionIndex];
                result[actionIndex + 1] = runningSum;
            }
            return result;
        }
        
        private static void ApplyDiscreteActionOutput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo,
            Multinomial multinomial,
            int[] actionSize)
        {
            var tensorDataProbabilities = tensor.Data as float[,];
            var batchSize = agentInfo.Keys.Count;
            var actions = new float[batchSize, actionSize.Length];
            var startActionIndices = CreateActionStartinIndices(actionSize);
            
            for (var actionIndex=0; actionIndex < actionSize.Length; actionIndex++)
            {
                var nBranchAction = startActionIndices[actionIndex + 1] -
                                    startActionIndices[actionIndex];
                var actionProbs = new float[batchSize, nBranchAction];
                for (var ii = 0; ii < batchSize; ii++)
                {
                    for (var jj = 0; jj < nBranchAction; jj++)
                    {
                        
                        actionProbs[ii, jj] = 
                            tensorDataProbabilities[ii, startActionIndices[actionIndex] + jj];
                    }
                }
                var inputTensor = new Tensor()
                {
                    ValueType= Tensor.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, actionSize[actionIndex]},
                    Data = actionProbs
                };
                var outputTensor = new Tensor()
                {
                    ValueType= Tensor.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, 1},
                    Data = new float[batchSize, 1]
                };
                multinomial.Eval(inputTensor, outputTensor);
                var outTensor = outputTensor.Data as float[,];
                for (var ii = 0; ii < batchSize; ii++)
                {
                    actions[ii, actionIndex] = outTensor[ii, 0];
                }
            }
            
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var a = new float[actionSize.Length];
                for (var j = 0; j < actionSize.Length; j++)
                {
                    a[j] = actions[agentIndex, j];
                }
    
                agent.UpdateVectorAction(a);
                agentIndex++;
            }
        }

        private static void ApplyMemoryOutput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataMemory = tensor.Data as float[,];
            var agentIndex = 0;
            var memorySize = tensor.Shape[1];
            foreach (var agent in agentInfo.Keys)
            {
                var a = new List<float>();
                for (var j = 0; j < memorySize; j++)
                {
                    a.Add(tensorDataMemory[agentIndex, j]);
                }

                // TODO Make better
                agent.UpdateMemoriesAction(a);
                agentIndex++;
            }
        }

        private static void ApplyValueEstimate(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataValue = tensor.Data as float[,];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                agent.UpdateValueAction(tensorDataValue[agentIndex, 0]);
                agentIndex++;
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
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.ObjectField(tfGraphModel);
            
            serializedBrain.ApplyModifiedProperties();
            
            if (EditorGUI.EndChangeCheck())
            {
                InitializeModel(m_model);
                tmp_editor_errors = GetModelErrors();
            }
            
            // TODO : Remove :
            tmp_editor_errors = GetModelErrors();
//            Debug.Log(GetModelErrors().Count);
            
            foreach (var error in tmp_editor_errors)
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