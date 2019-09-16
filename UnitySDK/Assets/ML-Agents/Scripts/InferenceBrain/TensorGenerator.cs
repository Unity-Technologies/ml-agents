using System.Collections.Generic;
using Barracuda;

namespace MLAgents.InferenceBrain
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
    public class TensorGenerator
    {
        public interface IGenerator
        {
            /// <summary>
            /// Modifies the data inside a Tensor according to the information contained in the
            /// AgentInfos contained in the current batch.
            /// </summary>
            /// <param name="tensorProxy"> The tensor the data and shape will be modified</param>
            /// <param name="batchSize"> The number of agents present in the current batch</param>
            /// <param name="agentInfo"> Dictionary of Agent to AgentInfo containing the
            /// information that will be used to populate the tensor's data</param>
            void Generate(
                TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo);
        }

        private readonly Dictionary<string, IGenerator> m_Dict = new Dictionary<string, IGenerator>();

        /// <summary>
        /// Returns a new TensorGenerators object.
        /// </summary>
        /// <param name="bp"> The BrainParameters used to determine what Generators will be
        /// used</param>
        /// <param name="seed"> The seed the Generators will be initialized with.</param>
        /// <param name="allocator"> Tensor allocator</param>
        /// <param name="barracudaModel"></param>
        public TensorGenerator(
            BrainParameters bp, int seed, ITensorAllocator allocator, object barracudaModel = null)
        {
            // Generator for Inputs
            m_Dict[TensorNames.BatchSizePlaceholder] =
                new BatchSizeGenerator(allocator);
            m_Dict[TensorNames.SequenceLengthPlaceholder] =
                new SequenceLengthGenerator(allocator);
            m_Dict[TensorNames.VectorObservationPlacholder] =
                new VectorObservationGenerator(allocator);
            m_Dict[TensorNames.RecurrentInPlaceholder] =
                new RecurrentInputGenerator(allocator);

            if (barracudaModel != null)
            {
                var model = (Model)barracudaModel;
                for (var i = 0; i < model?.memories.Length; i++)
                {
                    m_Dict[model.memories[i].input] =
                        new BarracudaRecurrentInputGenerator(i, allocator);
                }
            }

            m_Dict[TensorNames.PreviousActionPlaceholder] =
                new PreviousActionInputGenerator(allocator);
            m_Dict[TensorNames.ActionMaskPlaceholder] =
                new ActionMaskInputGenerator(allocator);
            m_Dict[TensorNames.RandomNormalEpsilonPlaceholder] =
                new RandomNormalInputGenerator(seed, allocator);
            if (bp.cameraResolutions != null)
            {
                for (var visIndex = 0;
                     visIndex < bp.cameraResolutions.Length;
                     visIndex++)
                {
                    var index = visIndex;
                    var bw = bp.cameraResolutions[visIndex].blackAndWhite;
                    m_Dict[TensorNames.VisualObservationPlaceholderPrefix + visIndex] =
                        new VisualObservationInputGenerator(index, bw, allocator);
                }
            }

            // Generators for Outputs
            m_Dict[TensorNames.ActionOutput] = new BiDimensionalOutputGenerator(allocator);
            m_Dict[TensorNames.RecurrentOutput] = new BiDimensionalOutputGenerator(allocator);
            m_Dict[TensorNames.ValueEstimateOutput] = new BiDimensionalOutputGenerator(allocator);
        }

        /// <summary>
        /// Populates the data of the tensor inputs given the data contained in the current batch
        /// of agents.
        /// </summary>
        /// <param name="tensors"> Enumerable of tensors that will be modified.</param>
        /// <param name="currentBatchSize"> The number of agents present in the current batch
        /// </param>
        /// <param name="agentInfos"> Dictionary of Agent to AgentInfo that contains the
        /// data that will be used to modify the tensors</param>
        /// <exception cref="UnityAgentsException"> One of the tensor does not have an
        /// associated generator.</exception>
        public void GenerateTensors(
            IEnumerable<TensorProxy> tensors,
            int currentBatchSize,
            Dictionary<Agent, AgentInfo> agentInfos)
        {
            foreach (var tensor in tensors)
            {
                if (!m_Dict.ContainsKey(tensor.name))
                {
                    throw new UnityAgentsException(
                        $"Unknown tensorProxy expected as input : {tensor.name}");
                }
                m_Dict[tensor.name].Generate(tensor, currentBatchSize, agentInfos);
            }
        }
    }
}
