using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MLAgents.InferenceBrain;
using UnityEngine.MachineLearning.InferenceEngine;
using UnityEngine.MachineLearning.InferenceEngine.Util;

namespace MLAgents
{
    /// <summary>
    /// The Learning Brain works differently if you are training it or not.
    /// When training your Agents, drag the Learning Brain to the Academy's BroadcastHub and check
    /// the checkbox Control. When using a pretrained model, just drag the Model file into the
    /// Model property of the Learning Brain.
    /// When the Learning Braom is noe training, it uses a TensorFlow model to make decisions.
    /// The Proximal Policy Optimization (PPO) and Behavioral Cloning algorithms included with
    /// the ML-Agents SDK produce trained TensorFlow models that you can use with the
    /// Learning Brain.
    /// </summary>
    [CreateAssetMenu(fileName = "NewLearningBrain", menuName = "ML-Agents/Learning Brain")]
    public class LearningBrain : Brain
    {
        private const long ApiVersion = 1; 
        private List<string> _failedModelChecks = new List<string>();
        private long _modelMemorySize;
        
        private TensorGenerators _tensorGenerators;
        private TensorAppliers  _outputTensorAppliers;
        
        public Model model;
        public InferenceEngineConfig.DeviceType inferenceDevice;

        InferenceEngine _engine;
        private IEnumerable<Tensor> _inferenceInputs;
        private IEnumerable<Tensor> _inferenceOutputs;

        private bool _isControlled;
        
        private double _inferenceDelta = 0;

        /// <summary>
        /// When Called, the brain will be controlled externally. It will not use the
        /// model to decide on actions.
        /// </summary>
        public void SetToControlledExternally()
        {
            _isControlled = true;
        }
        
        /// <inheritdoc />
        protected override void Initialize()
        {
            GiveModel(model);
        }
        
        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="newModel"> The model the brain will use</param>
        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
        /// and Multinomial obsjects used when running inference.</param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
        public void GiveModel(Model newModel, int seed = 0)
        {
            if (newModel != null)
            {
                var config = new InferenceEngineConfig
                {
                    Device = inferenceDevice
                };;
                _engine = InferenceAPI.LoadModel(newModel, config);

                var modelVersionNumber = GetModelData(_engine, TensorNames.VersionNumber);
                _modelMemorySize = GetModelData(_engine, TensorNames.MemorySize);
                var modelIsContinuous = GetModelData(_engine, TensorNames.IsContinuousControl);
                var modelActionSize =  GetModelData(_engine, TensorNames.ActionOutputShape);
                
                // Generate the Input tensors
                _inferenceInputs = GetInputTensors();
                _inferenceOutputs = GetOutputTensors();

                _failedModelChecks = TensorCheck.GetChecks(
                    _engine, 
                    _inferenceInputs, 
                    _inferenceOutputs, 
                    brainParameters,
                    modelVersionNumber,
                    ApiVersion,
                    modelIsContinuous, 
                    _modelMemorySize,
                    modelActionSize).ToList();
                
                _tensorGenerators = new TensorGenerators(
                    brainParameters, new RandomNormal(seed));
            
                _outputTensorAppliers = new TensorAppliers(
                    brainParameters, new Multinomial(seed));
            }
            else
            {
                _engine = null;
                _failedModelChecks = new List<string>();
                _failedModelChecks.Add(
                    "There is no model on this Brain, cannot run inference. (But can still train)");
            }
        }
        
        /// <summary>
        /// Return a list of failed checks corresponding to the failed compatibility checks
        /// between the Model and the BrainParameters. Note : This does not reload the model.
        /// If changes have been made to the BrainParameters or the Model, the model must be
        /// reloaded using GiveModel before trying to get the compatibility checks.
        /// </summary>
        /// <returns> The list of the failed compatibility checks between the Model and the
        /// Brain Parameters</returns>
        public List<string> GetModelFailedChecks()
        {
            return _failedModelChecks;
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
        private IEnumerable<Tensor> GetInputTensors()
        {
            return _engine.InputFeatures();
        }
        
        /// <summary>
        /// Generates the Tensor outputs that are expected to be present in the Model given the
        /// BrainParameters. 
        /// </summary>
        /// <returns>Tensor Array with the expected Tensor outputs</returns>
        private IEnumerable<Tensor> GetOutputTensors()
        {
            var bp = brainParameters;
            var tensorList = new List<Tensor>();
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.ActionOutput,
                    Shape = new long[]
                    {
                        -1, bp.vectorActionSize[0]
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null
                });
            }
            else
            {
                tensorList.Add(
                    new Tensor()
                    {
                        Name = TensorNames.ActionOutput,
                        Shape = new long[]
                        {
                            -1, bp.vectorActionSize.Sum()
                        },
                        ValueType = Tensor.TensorType.FloatingPoint,
                        Data = null
                    });
            }
            if (_modelMemorySize > 0)
            {

                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.RecurrentOutput,
                    Shape = new long[2]
                    {
                        -1, _modelMemorySize
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null
                });
            }
            return tensorList;
        }

        /// <inheritdoc />
        protected override void DecideAction()
        {
            base.DecideAction();
            if (_isControlled)
            {
                agentInfos.Clear();
                return;
            }
            var currentBatchSize = agentInfos.Count();
            if (currentBatchSize == 0)
            {
                return;
            }
            // Prepare the input tensors to be feed into the engine
            foreach (var tensor in _inferenceInputs)
            {
                if (!_tensorGenerators.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as input : "+tensor.Name);
                }
                _tensorGenerators[tensor.Name].Invoke(tensor, currentBatchSize, agentInfos);
            }
            // Prepare the output tensors to be feed into the engine
            foreach (var tensor in _inferenceOutputs)
            {
                if (!_tensorGenerators.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as output : "+tensor.Name);
                }
                
                _tensorGenerators[tensor.Name].Invoke(tensor, currentBatchSize, agentInfos);
            }

            // Execute the Model
            double startTime = Time.realtimeSinceStartup;
            _engine.ExecuteGraph(_inferenceInputs, _inferenceOutputs);
            _inferenceDelta = Time.realtimeSinceStartup - startTime;

            Debug.Log(_inferenceDelta * 1000);

            // Update the outputs
            foreach (var tensor in _inferenceOutputs)
            {
                if (!_outputTensorAppliers.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as output : "+tensor.Name);
                }
                _outputTensorAppliers[tensor.Name].Invoke(tensor, agentInfos);
            }
        }
    }
}
