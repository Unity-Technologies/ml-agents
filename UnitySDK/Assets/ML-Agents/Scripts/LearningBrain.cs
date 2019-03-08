using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MLAgents.InferenceBrain;
using UnityEngine.Profiling;

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
        private TensorGenerator _tensorGenerator;
        private TensorApplier _tensorApplier;
#if ENABLE_TENSORFLOW
        private ModelParamLoader _modelParamLoader;
#endif
        public TextAsset model;

#if ENABLE_TENSORFLOW
        private TFSharpInferenceEngine _engine;
#endif
        private IReadOnlyList<Tensor> _inferenceInputs;
        private IReadOnlyList<Tensor> _inferenceOutputs;

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
#if ENABLE_TENSORFLOW
            if (model != null)
            {
                _engine = new TFSharpInferenceEngine();
                _engine.PrepareModel(model.bytes);
            }
            else
            {
                _engine = null;
            }
            _modelParamLoader = ModelParamLoader.GetLoaderAndCheck(_engine, brainParameters);
            _inferenceInputs = _modelParamLoader.GetInputTensors();
            _inferenceOutputs = _modelParamLoader.GetOutputTensors();
            _tensorGenerator = new TensorGenerator(brainParameters, seed);
            _tensorApplier = new TensorApplier(brainParameters, seed);
#endif
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

#if ENABLE_TENSORFLOW
            return (_modelParamLoader != null) ? _modelParamLoader.GetChecks() : new List<string>();
#else
            return new List<string>(){
                "You need to install the TensorflowSharp plugin and add the ENABLE_TENSORFLOW " +
                "flag in your Player Settings in order to use inference. "};
#endif
        }

        /// <inheritdoc />
        protected override void DecideAction()
        {
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
#if ENABLE_TENSORFLOW
            if (_engine == null)
            {
                Debug.LogError($"No model was present for the Brain {name}.");
                return;
            }
            // Prepare the input tensors to be feed into the engine
            _tensorGenerator.GenerateTensors(_inferenceInputs, currentBatchSize, agentInfos);
            
            // Prepare the output tensors to be feed into the engine
            _tensorGenerator.GenerateTensors(_inferenceOutputs, currentBatchSize, agentInfos);

            // Execute the Model
            Profiler.BeginSample($"MLAgents.{name}.ExecuteGraph");
            _engine.ExecuteGraph(_inferenceInputs, _inferenceOutputs);
            Profiler.EndSample();

            // Update the outputs
            _tensorApplier.ApplyTensors(_inferenceOutputs, agentInfos);
#else
            if (agentInfos.Count > 0)
            {
                Debug.LogError(string.Format(
                    "The brain {0} was set to inference mode but the Tensorflow library is not " +
                    "present in the Unity project.",
                    name));
            }
#endif
            agentInfos.Clear();
        }
    }
}
