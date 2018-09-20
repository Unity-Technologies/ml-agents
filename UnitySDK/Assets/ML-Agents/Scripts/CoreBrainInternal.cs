using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.MachineLearning.InferenceEngine;
using UnityEngine.MachineLearning.InferenceEngine.Util;
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
        
        public class NodeNames
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

        private InternalBrainTensorGenerator _inputTensorGenerators;

        private InternalBrainTensorApplier  _outputTensorAppliers;

        public Model m_model;

        InferenceEngine m_engine;
        private Tensor[] inferenceInputs;
        private Tensor[] inferenceOutputs;

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

            
            _inputTensorGenerators = new InternalBrainTensorGenerator(
                _nodeNames, brain.brainParameters, new RandomNormal(0));
            
            _outputTensorAppliers = new InternalBrainTensorApplier(
                _nodeNames, brain.brainParameters, new Multinomial(0));

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

        public List<string> GetModelChecks()
        {
            if (m_engine == null)
            {
                return new List<string>();
            }
            
            // TODO : When the method is called, the engine and the brainParameters must be up to date
            inferenceInputs = GetInputTensors();
            inferenceOutputs = GetOutputTensors().ToArray();

            var failedChecks = new List<string>();
            if (_modelVersionNumber != version)
            {
                // TODO : Make better
                failedChecks.Add("Model has not been trained using the same version as the " +
                           "Internal Brain");
            }

            if (_isContinuous == 1 &&
                brain.brainParameters.vectorActionSpaceType != SpaceType.continuous)
            {
                // TODO : Implement
//                errors.Add("Error to implement");
            }
            failedChecks.AddRange(CheckInputTensorShape(
                inferenceInputs,
                brain.brainParameters,
                _nodeNames));
            failedChecks.AddRange(CheckInputTensorPresence(
                inferenceInputs,
                brain.brainParameters,
                _nodeNames));
            failedChecks.AddRange(CheckOutputTensorPresence(
                inferenceOutputs,
                brain.brainParameters,
                _nodeNames));
            return failedChecks;
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

        private static List<string> CheckInputTensorPresence(
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
        
        private static List<string> CheckOutputTensorPresence(
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

        private static List<string> CheckInputTensorShape(
            IEnumerable<Tensor> tensors, 
            BrainParameters brainParams,
            NodeNames nodeNames)
        {
            var result = new List<string>();
 
            var tensorTester =
                new Dictionary<string, Func<Tensor, BrainParameters, string>>()
                {
                    {nodeNames.VectorObservationPlacholder, CheckVectorObsShape},
                    {nodeNames.PreviousActionPlaceholder, CheckPreviousActionShape},
                    {nodeNames.RandomNormalEpsilonPlaceholder, ((tensor, parameters) => null)},
                    // TODO : Need a checker for some of those
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
                    (tensor, bp) => CheckVisualObsShape(tensor, bp, index);
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
        
        private static string CheckActionMaskShape(
            Tensor tensor,
            BrainParameters brainParams)
        {
            Debug.Log("error to implement.");
            return null;
        }

        // TODO : Rename these check
        private static string CheckVectorObsShape(
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
        
        private static string CheckPreviousActionShape(
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
        
        private static string CheckVisualObsShape(
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
                tmp_editor_errors = GetModelChecks();
            }
            
            // TODO : Remove :
            tmp_editor_errors = GetModelChecks();
//            Debug.Log(GetModelErrors().Count);
            
            foreach (var error in tmp_editor_errors)
            {
                if (error != null)
                    EditorGUILayout.HelpBox(error, MessageType.Error);
            }
#endif
        }

    }
}