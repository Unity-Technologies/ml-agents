using System;
using System.Collections.Generic;
using TransformsAI.MicroMLAgents.Sensors;
using Unity.Barracuda;

namespace TransformsAI.MicroMLAgents.Inference
{
    /// <summary>
    /// Mapping between Tensor names and generators.
    /// A TensorGenerator implements a Dictionary of strings (node names) to an Action.
    /// The Action take as argument the tensor, the current batch size and a Dictionary of
    /// Agent to AgentInfo corresponding to the current batch.
    /// Each Generator reshapes and fills the data of the tensor based of the data of the batch.
    /// When the TensorProxy is an Input to the model, the shape of the Tensor will be modified
    /// depending on the current batch size and the data of the Tensor will be filled using the
    /// Dictionary of Agent to AgentInfo.
    /// When the TensorProxy is an Output of the model, only the shape of the Tensor will be
    /// modified using the current batch size. The data will be pre-filled with zeros.
    /// </summary>
    internal class TensorGenerator
    {
        public interface IGenerator
        {
            /// <summary>
            /// Modifies the data inside a Tensor according to the information contained in the
            /// AgentInfos contained in the current batch.
            /// </summary>
            /// <param name="tensorProxy"> The tensor the data and shape will be modified.</param>
            /// <param name="batchSize"> The number of agents present in the current batch.</param>
            /// <param name="infos">
            /// List of AgentInfos containing the information that will be used to populate
            /// the tensor's data.
            /// </param>
            void Generate(TensorProxy tensorProxy, IList<IAgent> batch);
        }

        readonly Dictionary<string, IGenerator> m_Dict = new Dictionary<string, IGenerator>();

        /// <summary>
        /// Returns a new TensorGenerators object.
        /// </summary>
        /// <param name="seed"> The seed the Generators will be initialized with.</param>
        /// <param name="allocator"> Tensor allocator.</param>
        /// <param name="memories">Dictionary of AgentInfo.id to memory for use in the inference model.</param>
        /// <param name="barracudaModel"></param>
        /// <param name="deterministicInference"> Inference only: set to true if the action selection from model should be
        /// deterministic. </param>
        public TensorGenerator(
            int seed,
            IList<ObservationSpec> sensors,
            ITensorAllocator allocator,
            Model model,
            bool deterministicInference)
        {

            var apiVersion = model.GetVersion();
            
            if (apiVersion != (int) BarracudaModelParamLoader.ModelApiVersion.MLAgents2_0)
                throw new NotSupportedException("MLAgents Version Not Supported");

            // Generator for Inputs
            m_Dict[TensorNames.BatchSizePlaceholder] = new BatchSizeGenerator(allocator);
            m_Dict[TensorNames.SequenceLengthPlaceholder] = new SequenceLengthGenerator(allocator);
            m_Dict[TensorNames.RecurrentInPlaceholder] = new RecurrentInputGenerator(allocator);

            m_Dict[TensorNames.PreviousActionPlaceholder] = new PreviousActionInputGenerator(allocator);
            m_Dict[TensorNames.ActionMaskPlaceholder] = new ActionMaskInputGenerator(allocator);
            m_Dict[TensorNames.RandomNormalEpsilonPlaceholder] = new RandomNormalInputGenerator(seed, allocator);


            // Generators for Outputs
            if (model.HasContinuousOutputs(deterministicInference))
            {
                m_Dict[model.ContinuousOutputName(deterministicInference)] = new BiDimensionalOutputGenerator(allocator);
            }
            if (model.HasDiscreteOutputs(deterministicInference))
            {
                m_Dict[model.DiscreteOutputName(deterministicInference)] = new BiDimensionalOutputGenerator(allocator);
            }
            m_Dict[TensorNames.RecurrentOutput] = new BiDimensionalOutputGenerator(allocator);
            m_Dict[TensorNames.ValueEstimateOutput] = new BiDimensionalOutputGenerator(allocator);


            for (var sensorIndex = 0; sensorIndex < sensors.Count; sensorIndex++)
            {
                var obsGen = new ObservationGenerator(allocator,sensorIndex);
                var obsGenName = TensorNames.GetObservationName(sensorIndex);
                m_Dict[obsGenName] = obsGen;
            }
        }

        /// <summary>
        /// Populates the data of the tensor inputs given the data contained in the current batch
        /// of agents.
        /// </summary>
        /// <param name="tensors"> Enumerable of tensors that will be modified.</param>
        /// <param name="currentBatchSize"> The number of agents present in the current batch
        /// </param>
        /// <param name="infos"> List of AgentsInfos and Sensors that contains the
        /// data that will be used to modify the tensors</param>
        /// <exception cref="UnityAgentsException"> One of the tensor does not have an
        /// associated generator.</exception>
        public void GenerateTensors(IReadOnlyList<TensorProxy> tensors, IList<IAgent> infos)
        {
            for (var tensorIndex = 0; tensorIndex < tensors.Count; tensorIndex++)
            {
                var tensor = tensors[tensorIndex];
                if (!m_Dict.ContainsKey(tensor.name))
                {
                    throw new UnityAgentsException(
                        $"Unknown tensorProxy expected as input : {tensor.name}");
                }
                m_Dict[tensor.name].Generate(tensor, infos);
            }
        }
    }

}
