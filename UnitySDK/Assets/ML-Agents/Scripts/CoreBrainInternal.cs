//using System.Collections.Generic;
//using UnityEngine;
//using System.Linq;
//using UnityEngine.MachineLearning.InferenceEngine;
//using UnityEngine.MachineLearning.InferenceEngine.Util;
//using MLAgents.InferenceBrain;
//// TODO : Remove
//#if UNITY_EDITOR
//using UnityEditor;
//#endif
//
//namespace MLAgents
//{
//    /// CoreBrain which decides actions using internally embedded TensorFlow model.
//    public class CoreBrainInternal : ScriptableObject, CoreBrain
//    {
//        private const long ApiVersion = 1; 
//        
//        [SerializeField] [Tooltip("If checked, the brain will broadcast states and actions to Python.")]
//#pragma warning disable
//        private bool broadcast = true;
//#pragma warning restore
//
//
//        private List<string> _failedModelChecks = new List<string>();
//        
//        private Batcher brainBatcher;
//        
//        private long _modelMemorySize;
//
//        private TensorGenerators _tensorGenerators;
//        private TensorAppliers  _outputTensorAppliers;
//
//        private Model _model;
//
//        InferenceEngine _engine;
//        private IEnumerable<Tensor> _inferenceInputs;
//        private IEnumerable<Tensor> _inferenceOutputs;
//
//        public Brain brain;
//
//        /// Create the reference to the brain
//        public void SetBrain(Brain b)
//        {
//            brain = b;
//        }
//
//        /// Loads the tensorflow graph model to generate a TFGraph object
//        public void InitializeCoreBrain(Batcher brainBatcher)
//        {
//            if ((brainBatcher == null)
//                || (!broadcast))
//            {
//                this.brainBatcher = null;
//            }
//            else
//            {
//                this.brainBatcher = brainBatcher;
//                this.brainBatcher.SubscribeBrain(brain.gameObject.name);
//            }
//
//            GiveModel(_model);
//
//        }
//
//        /// <summary>
//        /// Initializes the Brain with the Model that it will use when selecting actions for
//        /// the agents
//        /// </summary>
//        /// <param name="model"> The model the brain will use</param>
//        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
//        /// and Multinomial obsjects used when running inference.</param>
//        /// <exception cref="UnityAgentsException">Throws an error when the model is null
//        /// </exception>
//        public void GiveModel(Model model, int seed = 0)
//        {
//            if (model != null)
//            {
//                InferenceEngineConfig config;
//                _engine = InferenceAPI.LoadModel(model, config);
//
//                var modelVersionNumber = GetModelData(_engine, TensorNames.VersionNumber);
//                _modelMemorySize = GetModelData(_engine, TensorNames.MemorySize);
//                var modelIsContinuous = GetModelData(_engine, TensorNames.IsContinuousControl);
//                var modelActionSize =  GetModelData(_engine, TensorNames.ActionOutputShape);
//                
//                // Generate the Input tensors
//                _inferenceInputs = GetInputTensors();
//                _inferenceOutputs = GetOutputTensors();
//
//                _failedModelChecks = TensorCheck.GetChecks(_engine, 
//                    _inferenceInputs, 
//                    _inferenceOutputs, 
//                    brain.brainParameters,
//                    modelVersionNumber,
//                    ApiVersion,
//                    modelIsContinuous, 
//                    _modelMemorySize,
//                    modelActionSize).ToList();
//                
//                _tensorGenerators = new TensorGenerators(brain.brainParameters, 
//                    new RandomNormal(seed));
//            
//                _outputTensorAppliers = new TensorAppliers(brain.brainParameters, 
//                    new Multinomial(seed));
//            }
//            else
//            {
//                _engine = null;
//                _failedModelChecks = new List<string>();
//                _failedModelChecks.Add(
//                    "There is no model on this Brain, cannot run inference. (But can still train)");
//                // TODO : Implement
//                throw new UnityAgentsException("ERROR TO IMPLEMENT : There was no model");
//            }
//        }
//
//        /// <summary>
//        /// Return a list of failed checks corresponding to the failed compatibility checks
//        /// between the Model and the BrainParameters. Note : This does not reload the model.
//        /// If changes have been made to the BrainParameters or the Model, the model must be
//        /// reloaded using GiveModel before trying to get the compatibility checks.
//        /// </summary>
//        /// <returns> The list of the failed compatibility checks between the Model and the
//        /// Brain Parameters</returns>
//        public List<string> GetModelFailedChecks()
//        {
//            return _failedModelChecks;
//        }
//
//        /// <summary>
//        /// Queries the InferenceEngine for the value of a variable in the graph given its name.
//        /// Only works with int32 Tensors with zero dimensions containing a unique element.
//        /// If the node was not found or could not be retrieved, the value -1 will be returned. 
//        /// </summary>
//        /// <param name="engine">The InferenceEngine to be queried</param>
//        /// <param name="name">The name of the Tensor variable</param>
//        /// <returns></returns>
//        private static long GetModelData(InferenceEngine engine, string name)
//        {
//            try
//            {
//                var outputs = new Tensor[]
//                {
//                    new Tensor()
//                    {
//                        Name = name,
//                        ValueType = Tensor.TensorType.Integer,
//                        Shape = new long[]{},
//                        Data = new long[1]
//                    },
//                };
//                engine.ExecuteGraph(new Tensor[0], outputs);
//           
//                return (outputs[0].Data as int[])[0];
//            }
//            catch
//            {
//                Debug.Log("Node not in graph " + name);
//                return -1;
//            }
//        }
//
//        /// <summary>
//        /// Generates the Tensor inputs that are expected to be present in the Model given the
//        /// BrainParameters. 
//        /// </summary>
//        /// <returns>Tensor Array with the expected Tensor inputs</returns>
//        private IEnumerable<Tensor> GetInputTensors()
//        {
//            return _engine.InputFeatures();
//        }
//        
//        /// <summary>
//        /// Generates the Tensor outputs that are expected to be present in the Model given the
//        /// BrainParameters. 
//        /// </summary>
//        /// <returns>Tensor Array with the expected Tensor outputs</returns>
//        private IEnumerable<Tensor> GetOutputTensors()
//        {
//            var bp = brain.brainParameters;
//            var tensorList = new List<Tensor>();
//            if (bp.vectorActionSpaceType == SpaceType.continuous)
//            {
//                tensorList.Add(new Tensor()
//                {
//                    Name = TensorNames.ActionOutput,
//                    Shape = new long[]
//                    {
//                        -1, bp.vectorActionSize[0]
//                    },
//                    ValueType = Tensor.TensorType.FloatingPoint,
//                    Data = null
//                });
//            }
//            else
//            {
//                tensorList.Add(
//                    new Tensor()
//                    {
//                        Name = TensorNames.ActionOutput,
//                        Shape = new long[]
//                        {
//                            -1, bp.vectorActionSize.Sum()
//                        },
//                        ValueType = Tensor.TensorType.FloatingPoint,
//                        Data = null
//                    });
//            }
//            if (_modelMemorySize > 0)
//            {
//
//                tensorList.Add(new Tensor()
//                {
//                    Name = TensorNames.RecurrentOutput,
//                    Shape = new long[2]
//                    {
//                        -1, _modelMemorySize
//                    },
//                    ValueType = Tensor.TensorType.FloatingPoint,
//                    Data = null
//                });
//            }
//
//            return tensorList;
//
//        }
//
//        /// Uses the stored information to run the tensorflow graph and generate 
//        /// the actions.
//        public void DecideAction(Dictionary<Agent, AgentInfo> agentInfo)
//        {
//            if (brainBatcher != null)
//            {
//                brainBatcher.SendBrainInfo(brain.gameObject.name, agentInfo);
//            }
//
//            var currentBatchSize = agentInfo.Count();
//            if (currentBatchSize == 0)
//            {
//                return;
//            }
//
//            foreach (var tensor in _inferenceInputs)
//            {
//                if (!_tensorGenerators.ContainsKey(tensor.Name))
//                {
//                    throw new UnityAgentsException("Error to implement.");
//                }
//                _tensorGenerators[tensor.Name].Invoke(tensor, currentBatchSize, agentInfo);
//            }
//            
//            foreach (var tensor in _inferenceOutputs)
//            {
//                if (!_tensorGenerators.ContainsKey(tensor.Name))
//                {
//                    throw new UnityAgentsException("Error to implement.");
//                }
//                
//                _tensorGenerators[tensor.Name].Invoke(tensor, currentBatchSize, agentInfo);
//            }
//
//            // Execute the Model
//            _engine.ExecuteGraph(_inferenceInputs, _inferenceOutputs);
//
//            // Update the outputs
//            foreach (var tensor in _inferenceOutputs)
//            {
//                if (!_outputTensorAppliers.ContainsKey(tensor.Name))
//                {
//                    throw new UnityAgentsException("Error to implement.");
//                }
//                _outputTensorAppliers[tensor.Name].Invoke(tensor, agentInfo);
//            }
//        }
//        
//        /// Displays the parameters of the CoreBrainInternal in the Inspector 
//        public void OnInspector()
//        {
//#if UNITY_EDITOR
//            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
//            broadcast = EditorGUILayout.Toggle(new GUIContent("Broadcast",
//                "If checked, the brain will broadcast states and actions to Python."), broadcast);
//
//            var serializedBrain = new SerializedObject(this);
//            GUILayout.Label("Edit the Tensorflow graph parameters here");
//            var tfGraphModel = serializedBrain.FindProperty("_model");
//            serializedBrain.Update();
//            EditorGUI.BeginChangeCheck();
//            EditorGUILayout.ObjectField(tfGraphModel);
//            
//            serializedBrain.ApplyModifiedProperties();
//            
//            if (EditorGUI.EndChangeCheck())
//            {
//                GiveModel(_model);
//            }
//
//            if (GUILayout.Button("Check model"))
//            {
//                GiveModel(_model);
//            }
//
//            foreach (var error in GetModelFailedChecks())
//            {
//                if (error != null)
//                    EditorGUILayout.HelpBox(error, MessageType.Warning);
//            }
//#endif
//        }
//
//    }
//}
