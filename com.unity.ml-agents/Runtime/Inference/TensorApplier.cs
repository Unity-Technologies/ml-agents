using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;


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
    internal class TensorApplier
    {
        /// <summary>
        /// A tensor Applier's Execute method takes a tensor and a Dictionary of Agent to AgentInfo.
        /// Uses the data contained inside the tensor to modify the state of the Agent. The Tensors
        /// are assumed to have the batch size on the first dimension and the agents to be ordered
        /// the same way in the dictionary and in the tensor.
        /// </summary>
        public interface IApplier
        {
            /// <summary>
            /// Applies the values in the Tensor to the Agents present in the agentInfos
            /// </summary>
            /// <param name="tensorProxy">
            /// The Tensor containing the data to be applied to the Agents
            /// </param>
            /// <param name="actionIds"> List of Agents Ids that will be updated using the tensor's data</param>
            /// <param name="lastActions"> Dictionary of AgentId to Actions to be updated</param>
            void Apply(TensorProxy tensorProxy, IList<int> actionIds, Dictionary<int, ActionBuffers> lastActions);
        }

        readonly Dictionary<string, IApplier> m_Dict = new Dictionary<string, IApplier>();

        /// <summary>
        /// Returns a new TensorAppliers object.
        /// </summary>
        /// <param name="actionSpec"> Description of the actions for the Agent.</param>
        /// <param name="seed"> The seed the Appliers will be initialized with.</param>
        /// <param name="allocator"> Tensor allocator</param>
        /// <param name="memories">Dictionary of AgentInfo.id to memory used to pass to the inference model.</param>
        /// <param name="barracudaModel"></param>
        public TensorApplier(
            ActionSpec actionSpec,
            int seed,
            ITensorAllocator allocator,
            Dictionary<int, List<float>> memories,
            object barracudaModel = null)
        {
            // If model is null, no inference to run and exception is thrown before reaching here.
            if (barracudaModel == null)
            {
                return;
            }

            var model = (Model)barracudaModel;
            if (!model.SupportsContinuousAndDiscrete())
            {
                actionSpec.CheckAllContinuousOrDiscrete();
            }
            if (actionSpec.NumContinuousActions > 0)
            {
                var tensorName = model.ContinuousOutputName();
                m_Dict[tensorName] = new ContinuousActionOutputApplier(actionSpec);
            }
            var modelVersion = model.GetVersion();
            if (actionSpec.NumDiscreteActions > 0)
            {
                var tensorName = model.DiscreteOutputName();
                if (modelVersion == (int)BarracudaModelParamLoader.ModelApiVersion.MLAgents1_0)
                {
                    m_Dict[tensorName] = new LegacyDiscreteActionOutputApplier(actionSpec, seed, allocator);
                }
                if (modelVersion == (int)BarracudaModelParamLoader.ModelApiVersion.MLAgents2_0)
                {
                    m_Dict[tensorName] = new DiscreteActionOutputApplier(actionSpec, seed, allocator);
                }
            }
            m_Dict[TensorNames.RecurrentOutput] = new MemoryOutputApplier(memories);
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
}
