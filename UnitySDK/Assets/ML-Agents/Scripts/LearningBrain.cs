using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MLAgents.InferenceBrain;
using UnityEngine.MachineLearning.InferenceEngine;

namespace MLAgents
{
    /// <summary>
    /// The Learning Brain works differently if you are training it or not.
    /// When training your Agents, drag the Learning Brain to the Academy's BroadcastHub and check
    /// the checkbox Control. When using a pretrained model, just drag the Model file into the
    /// Model property of the Learning Brain.
    /// The property model corresponds to the Model currently attached to the Brain. Before
    /// being used, a call to ReloadModel is required.
    /// When the Learning Brain is not training, it uses a TensorFlow model to make decisions.
    /// The Proximal Policy Optimization (PPO) and Behavioral Cloning algorithms included with
    /// the ML-Agents SDK produce trained TensorFlow models that you can use with the
    /// Learning Brain.
    /// </summary>
    [CreateAssetMenu(fileName = "NewLearningBrain", menuName = "ML-Agents/Learning Brain")]
    public class LearningBrain : Brain
    {
        private TensorGeneratorInvoker _tensorGeneratorInvoker;
        private TensorApplierInvoker _outputTensorApplierInvoker;
        private MetaDataLoader _metaDataLoader;
        
        public Model model;
        public InferenceEngineConfig.DeviceType inferenceDevice;

        private InferenceEngine _engine;
        private IEnumerable<Tensor> _inferenceInputs;
        private IEnumerable<Tensor> _inferenceOutputs;

        [NonSerialized]
        private bool _isControlled;

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
            ReloadModel();
        }
        
        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
        /// and Multinomial obsjects used when running inference.</param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
        public void ReloadModel(int seed = 0)
        {
            if (model != null)
            {
                var config = new InferenceEngineConfig
                {
                    Device = inferenceDevice
                };
                _engine = InferenceAPI.LoadModel(model, config);
            }
            else
            {
                _engine = null;
            }
            _metaDataLoader = new MetaDataLoader(_engine, brainParameters);
            _inferenceInputs = _metaDataLoader.GetInputTensors();
            _inferenceOutputs = _metaDataLoader.GetOutputTensors();
            _tensorGeneratorInvoker = new TensorGeneratorInvoker(brainParameters, seed);
            _outputTensorApplierInvoker = new TensorApplierInvoker(brainParameters, seed);
        }
        
        /// <summary>
        /// Return a list of failed checks corresponding to the failed compatibility checks
        /// between the Model and the BrainParameters. Note : This does not reload the model.
        /// If changes have been made to the BrainParameters or the Model, the model must be
        /// reloaded using GiveModel before trying to get the compatibility checks.
        /// </summary>
        /// <returns> The list of the failed compatibility checks between the Model and the
        /// Brain Parameters</returns>
        public IEnumerable<string> GetModelFailedChecks()
        {
            return (_metaDataLoader != null) ? _metaDataLoader.GetChecks() : new List<string>();
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
                if (!_tensorGeneratorInvoker.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as input : "+tensor.Name);
                }
                _tensorGeneratorInvoker[tensor.Name].Execute(tensor, currentBatchSize, agentInfos);
            }
            // Prepare the output tensors to be feed into the engine
            foreach (var tensor in _inferenceOutputs)
            {
                if (!_tensorGeneratorInvoker.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as output : "+tensor.Name);
                }
                
                _tensorGeneratorInvoker[tensor.Name].Execute(tensor, currentBatchSize, agentInfos);
            }

            // Execute the Model
            _engine.ExecuteGraph(_inferenceInputs, _inferenceOutputs);

            // Update the outputs
            foreach (var tensor in _inferenceOutputs)
            {
                if (!_outputTensorApplierInvoker.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as output : "+tensor.Name);
                }
                _outputTensorApplierInvoker[tensor.Name].Execute(tensor, agentInfos);
            }
            agentInfos.Clear();
        }
    }
}
