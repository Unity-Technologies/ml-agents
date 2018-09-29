using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.MachineLearning.InferenceEngine;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using MLAgents.InferenceBrain;
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
        
        [SerializeField] [Tooltip("If checked, the brain will broadcast states and actions to Python.")]
#pragma warning disable
        private bool broadcast = true;
#pragma warning restore


        public List<string> currentFailedModelChecks = new List<string>();
        
        Batcher brainBatcher;
        
        private long _memorySize;
        private long _modelVersionNumber;
        private long _isContinuous;

        private TensorGenerators _tensorGenerators;

        private TensorAppliers  _outputTensorAppliers;

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

            
            _tensorGenerators = new TensorGenerators(brain.brainParameters, new RandomNormal(0));
            
            _outputTensorAppliers = new TensorAppliers(brain.brainParameters, new Multinomial(0));

        }

        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="model">The model the brain will use</param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
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

                _modelVersionNumber = GetModelData(m_engine, TensorNames.VersionNumber);
                _memorySize = GetModelData(m_engine, TensorNames.MemorySize);
                _isContinuous = GetModelData(m_engine, TensorNames.IsContinuousControl);

                currentFailedModelChecks = new List<string>();
                if (_modelVersionNumber != version)
                {
                    currentFailedModelChecks.Add("Incompatible Version");
                }
                if (_memorySize == -1)
                {
                    currentFailedModelChecks.Add("No Memory Size");
                }
                if (_isContinuous == -1)
                {
                    currentFailedModelChecks.Add("Could not infer action space type from model");
                }
                currentFailedModelChecks.AddRange(TensorCheck.GetChecks(m_engine, 
                    inferenceInputs, 
                    inferenceOutputs, 
                    brain.brainParameters,
                    _isContinuous, 
                    _memorySize > 0));
            }
            else
            {
                m_engine = null;
                currentFailedModelChecks = new List<string>();
                currentFailedModelChecks.Add(
                    "There is no model on this Brain, cannot run inference. (But can still train)");
                // TODO : Implement
                throw new UnityAgentsException("ERROR TO IMPLEMENT : There was no model");
            }
        }

        /// <summary>
        /// Queries the InferenceEngine for the value of a variable in the graph given its name.
        /// Only works with int32 Tensors with zero dimensions containing a unique element.
        /// If the node was not found or could not be retrieved, the value -1 will be returned. 
        /// </summary>
        /// <param name="engine">The InferenceEngine to be queried</param>
        /// <param name="name">The name of the Tensor variable</param>
        /// <returns></returns>
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
                Debug.Log("Node not in graph " + name);
                return -1;
            }
        }

        /// <summary>
        /// Generates the Tensor inputs that are expected to be present in the Model given the
        /// BrainParameters. 
        /// </summary>
        /// <returns>Tensor Array with the expected Tensor inputs</returns>
        private Tensor[] GetInputTensors()
        {
            var n_agents = -1;
            BrainParameters bp = brain.brainParameters;

            var tensorList = new List<Tensor>();
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorList.Add(
                    new Tensor()
                {
                    Name = TensorNames.RandomNormalEpsilonPlaceholder,
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
                    Name = TensorNames.ActionMaskPlaceholder,
                    Shape = new long[]{n_agents, bp.vectorActionSize.Sum()},
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = new float[n_agents, bp.vectorActionSize.Sum()]
                });
            }
            foreach(var res in bp.cameraResolutions)
            {
                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.VisualObservationPlaceholderPrefix + "0",
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
                      Name = TensorNames.VectorObservationPlacholder,
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
                    Name = TensorNames.RecurrentInPlaceholder,
                    Shape = new long[2]
                    {
                        n_agents, _memorySize
                    },
                    ValueType = Tensor.TensorType.FloatingPoint
                });
                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.SequenceLengthPlaceholder,
                    Shape = new long[]
                    {
                        
                    },
                    ValueType = Tensor.TensorType.Integer
                });
                if (brain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
                {
                    tensorList.Add(new Tensor()
                    {
                        Name = TensorNames.PreviousActionPlaceholder,
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
        
        /// <summary>
        /// Generates the Tensor outputs that are expected to be present in the Model given the
        /// BrainParameters. 
        /// </summary>
        /// <returns>Tensor Array with the expected Tensor outputs</returns>
        private List<Tensor> GetOutputTensors()
        {
            var n_agents = -1;
            BrainParameters bp = brain.brainParameters;
            var tensorList = new List<Tensor>();
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.ActionOutput,
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
                        Name = TensorNames.ActionOutput,
                        Shape = new long[]{n_agents, bp.vectorActionSize.Sum()},
                        ValueType = Tensor.TensorType.FloatingPoint,
                        Data = new float[n_agents, bp.vectorActionSize.Sum()]
                    });
            }
            if (_memorySize > 0)
            {
                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.RecurrentOutOutput,
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
                if (!_tensorGenerators.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException("Error to implement.");
                }
                _tensorGenerators[tensor.Name].Invoke(tensor, currentBatchSize, agentInfo);

            }
            
            // Generating the Output tensors
            for (var tensorIndex = 0; tensorIndex<inferenceOutputs.Length; tensorIndex++)
            {
                var tensor = inferenceOutputs[tensorIndex];
                if (!_tensorGenerators.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException("Error to implement.");
                }
                _tensorGenerators[tensor.Name].Invoke(tensor, currentBatchSize, agentInfo);

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
            }
            
            // TODO : Remove :
//            Debug.Log(GetModelErrors().Count);
            
            foreach (var error in currentFailedModelChecks)
            {
                if (error != null)
                    EditorGUILayout.HelpBox(error, MessageType.Error);
            }
#endif
        }

    }
}