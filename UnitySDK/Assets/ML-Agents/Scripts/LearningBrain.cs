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
    public class LearningBrain : IBrain
    {

        private string m_BehaviorName;

        private BrainParameters m_BrainParameters;

        [Tooltip("Inference execution device. CPU is the fastest option for most of ML Agents models. " +
            "(This field is not applicable for training).")]

        protected IBatchedDecisionMaker m_BatchedDecisionMaker;

        /// <inheritdoc />
        public LearningBrain(
            BrainParameters brainParameters,
            NNModel model,
            InferenceDevice inferenceDevice,
            string behaviorName)
        {
            m_BrainParameters = brainParameters;
            m_BehaviorName = behaviorName;
            var aca = GameObject.FindObjectOfType<Academy>();
            aca.LazyInitialization();
            var comm = aca?.Communicator;
            SetCommunicator(comm);
            if (aca == null || comm != null)
            {
                return;
            }
            var modelRunner = aca.GetOrCreateModelRunner(model, brainParameters, inferenceDevice);
            m_BatchedDecisionMaker = modelRunner;
        }

        /// <summary>
        /// Sets the ICommunicator of the Brain. The brain will call the communicator at every step and give
        /// it the agent's data using PutObservations at each DecideAction call.
        /// </summary>
        /// <param name="communicator"> The Batcher the brain will use for the current session</param>
        private void SetCommunicator(ICommunicator communicator)
        {
            m_BatchedDecisionMaker = communicator;
            communicator?.SubscribeBrain(m_BehaviorName, m_BrainParameters);
        }

        /// <inheritdoc />
        public void RequestDecision(Agent agent)
        {
            m_BatchedDecisionMaker?.PutObservations(m_BehaviorName, agent);
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
