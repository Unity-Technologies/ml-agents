using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using System.Linq;
using Unity.MLAgents.Inference.Utils;
using UnityEngine;



namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Mapping between the output tensor names and the method that will use the
    /// output tensors and the Agents present in the batch to update their action, memories and
    /// value estimates.
    /// A TensorApplier implements a Dictionary of strings (node names) to an Action.
    /// This action takes as input the tensor and the Dictionary of Agent to AgentInfo for
    /// the current batch.
    /// </summary>
    internal class TrainingForwardTensorApplier
    {

        readonly Dictionary<string, TensorApplier.IApplier> m_Dict = new Dictionary<string, TensorApplier.IApplier>();

        /// <summary>
        /// Returns a new TensorAppliers object.
        /// </summary>
        /// <param name="actionSpec"> Description of the actions for the Agent.</param>
        /// <param name="seed"> The seed the Appliers will be initialized with.</param>
        /// <param name="allocator"> Tensor allocator</param>
        /// <param name="memories">Dictionary of AgentInfo.id to memory used to pass to the inference model.</param>
        /// <param name="barracudaModel"></param>
        public TrainingForwardTensorApplier(
            ActionSpec actionSpec,
            int seed,
            ITensorAllocator allocator,
            object barracudaModel = null)
        {
            // If model is null, no inference to run and exception is thrown before reaching here.
            if (barracudaModel == null)
            {
                return;
            }
            if (actionSpec.NumContinuousActions > 0)
            {
                throw new System.Exception("Cannot do continuous actions");
            }
            if (actionSpec.NumDiscreteActions != 1)
            {
                throw new System.Exception("Cannot do multi discrete actions, only single discrete");
            }

            var model = (Model)barracudaModel;


            m_Dict[TensorNames.TrainingOutput] = new MaxActionOutputApplier(actionSpec, seed, allocator);
        }

        /// <summary>
        /// Updates the state of the agents based on the data present in the tensor.
        /// </summary>
        /// <param name="tensors"> Enumerable of tensors containing the data.</param>
        /// <param name="actionIds"> List of Agents Ids that will be updated using the tensor's data</param>
        /// <param name="lastActions"> Dictionary of AgentId to Actions to be updated</param>
        /// <exception cref="UnityAgentsException"> One of the tensor does not have an
        /// associated applier.</exception>
        public void ApplyTensors(
            IReadOnlyList<TensorProxy> tensors, IList<int> actionIds, Dictionary<int, ActionBuffers> lastActions)
        {
            for (var tensorIndex = 0; tensorIndex < tensors.Count; tensorIndex++)
            {
                var tensor = tensors[tensorIndex];
                if (!m_Dict.ContainsKey(tensor.name))
                {
                    throw new UnityAgentsException(
                        $"Unknown tensorProxy expected as output : {tensor.name}");
                }
                m_Dict[tensor.name].Apply(tensor, actionIds, lastActions);
            }
        }

    }

    internal class MaxActionOutputApplier : TensorApplier.IApplier
    {
        readonly ActionSpec m_ActionSpec;


        public MaxActionOutputApplier(ActionSpec actionSpec, int seed, ITensorAllocator allocator)
        {
            m_ActionSpec = actionSpec;
        }

        public void Apply(TensorProxy tensorProxy, IList<int> actionIds, Dictionary<int, ActionBuffers> lastActions)
        {
            var agentIndex = 0;
            var actionSpaceSize = tensorProxy.shape[tensorProxy.shape.Length - 1];

            for (var i = 0; i < actionIds.Count; i++)
            {
                var agentId = actionIds[i];
                if (lastActions.ContainsKey(agentId))
                {
                    var actionBuffer = lastActions[agentId];
                    if (actionBuffer.IsEmpty())
                    {
                        actionBuffer = new ActionBuffers(m_ActionSpec);
                        lastActions[agentId] = actionBuffer;
                    }
                    var discreteBuffer = actionBuffer.DiscreteActions;
                    var maxIndex = 0;
                    var maxValue = 0;
                    for (var j = 0; j < actionSpaceSize; j++)
                    {
                        var value = (int)tensorProxy.data[agentIndex, j];
                        if (value > maxValue)
                        {
                            maxIndex = j;
                        }
                    }
                    var actionSize = discreteBuffer.Length;
                    discreteBuffer[0] = maxIndex;
                }
                agentIndex++;
            }
        }

    }

}
