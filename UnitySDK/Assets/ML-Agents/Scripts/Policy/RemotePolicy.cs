using UnityEngine;
using Barracuda;
using MLAgents.InferenceBrain;

namespace MLAgents
{
    /// <summary>
    /// The Remote Policy only works when training.
    /// When training your Agents, the RemotePolicy will be controlled by Python.
    /// </summary>
    public class RemotePolicy : IPolicy
    {

        private string m_BehaviorName;

        [Tooltip("Inference execution device. CPU is the fastest option for most of ML Agents models. " +
            "(This field is not applicable for training).")]

        protected IBatchedDecisionMaker m_BatchedDecisionMaker;

        /// <inheritdoc />
        public RemotePolicy(
            BrainParameters brainParameters,
            string behaviorName)
        {
            m_BehaviorName = behaviorName;
            var aca = GameObject.FindObjectOfType<Academy>();
            aca.LazyInitialization();
            m_BatchedDecisionMaker = aca.Communicator;
            aca.Communicator.SubscribeBrain(m_BehaviorName, brainParameters);
        }

        /// <inheritdoc />
        public void RequestDecision(Agent agent)
        {
            m_BatchedDecisionMaker?.PutObservations(m_BehaviorName, agent);
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
