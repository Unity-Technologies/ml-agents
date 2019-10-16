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
    /// The Barracuda Policy uses a Barracuda Model to make decisions at
    /// every step. It uses a ModelRunner that is shared accross all
    /// Barracuda Policies that use the same model and inference devices.
    /// </summary>
    public class BarracudaPolicy : IPolicy
    {

        protected IBatchedDecisionMaker m_BatchedDecisionMaker;

        /// <inheritdoc />
        public BarracudaPolicy(
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

        /// <inheritdoc />
        public void DecideAction()
        {
            m_BatchedDecisionMaker?.DecideBatch();
        }

        public void Dispose()
        {
        }
    }
}
