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
    [CreateAssetMenu(fileName = "NewLearningBrain", menuName = "ML-Agents/Learning Brain")]
    public class LearningBrain : Brain
    {
        public NNModel model;

        [Tooltip("Inference execution device. CPU is the fastest option for most of ML Agents models. " +
            "(This field is not applicable for training).")]
        public InferenceDevice inferenceDevice = InferenceDevice.CPU;

        protected IBatchedDecisionMaker m_BatchedDecisionMaker;

        /// <summary>
        /// Sets the ICommunicator of the Brain. The brain will call the communicator at every step and give
        /// it the agent's data using PutObservations at each DecideAction call.
        /// </summary>
        /// <param name="communicator"> The Batcher the brain will use for the current session</param>
        private void SetCommunicator(ICommunicator communicator)
        {
            m_BatchedDecisionMaker = communicator;
            communicator?.SubscribeBrain(name, brainParameters);
            LazyInitialize();

        }

        /// <inheritdoc />
        protected override void Initialize()
        {
            var aca = FindObjectOfType<Academy>();
            var comm = aca?.Communicator;
            SetCommunicator(comm);
            if (aca == null || comm != null)
            {
                return;
            }
            var modelRunner = aca.ModelRunners.Find(x => x.HasModel(model));
            if (modelRunner == null)
            {
                modelRunner = new BrainModelRunner(
                    model, brainParameters, inferenceDevice);
                aca.ModelRunners.Add(modelRunner);
            }
            m_BatchedDecisionMaker = modelRunner;
        }

        /// <inheritdoc />
        protected override void DecideAction()
        {
            if (m_BatchedDecisionMaker != null)
            {
                m_BatchedDecisionMaker?.PutObservations(name, m_Agents);
                return;
            }
        }

        public void OnDisable()
        {
            m_BatchedDecisionMaker?.Dispose();
        }
    }
}
