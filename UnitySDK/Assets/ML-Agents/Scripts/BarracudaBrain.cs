using UnityEngine;
using Barracuda;
using MLAgents.InferenceBrain;

namespace MLAgents
{
    public enum InferenceDevice
    {
        CPU = 0,
        GPU = 1
    }

    /// <summary>
    /// The Learning Brain works differently if you are training it or not.
    /// When training your Agents, the LearningBrain will be controlled by Python.
    /// When using a pretrained model, just drag the Model file into the
    /// Model property of the Learning Brain and do not launch the Python training process.
    /// The training will start automatically if Python is ready to train and there is at
    /// least one LearningBrain in the scene.
    /// The property model corresponds to the Model currently attached to the Brain. Before
    /// being used, a call to ReloadModel is required.
    /// When the Learning Brain is not training, it uses a TensorFlow model to make decisions.
    /// The Proximal Policy Optimization (PPO) and Behavioral Cloning algorithms included with
    /// the ML-Agents SDK produce trained TensorFlow models that you can use with the
    /// Learning Brain.
    /// </summary>
    public class BarracudaBrain : IBrain
    {

        [Tooltip("Inference execution device. CPU is the fastest option for most of ML Agents models. " +
            "(This field is not applicable for training).")]

        protected IBatchedDecisionMaker m_BatchedDecisionMaker;

        /// <inheritdoc />
        public BarracudaBrain(
            BrainParameters brainParameters,
            NNModel model,
            InferenceDevice inferenceDevice)
        {
            var aca = GameObject.FindObjectOfType<Academy>();
            aca.LazyInitialization();
            var modelRunner = aca.GetOrCreateModelRunner(model, brainParameters, inferenceDevice);
            m_BatchedDecisionMaker = modelRunner;
        }

        /// <inheritdoc />
        public void RequestDecision(Agent agent)
        {
            m_BatchedDecisionMaker?.PutObservations(null, agent);
        }

        public void DecideAction()
        {
            m_BatchedDecisionMaker?.DecideBatch();
        }

        public void Dispose()
        {
        }
    }
}
