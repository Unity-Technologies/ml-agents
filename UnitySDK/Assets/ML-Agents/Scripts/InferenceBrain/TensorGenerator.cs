#define ENABLE_BARRACUDA
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
    /// When the Tensor is an Input to the model, the shape of the Tensor will be modified
    /// depending on the current batch size and the data of the Tensor will be filled using the
    /// Dictionary of Agent to AgentInfo.
    /// When the Tensor is an Output of the model, only the shape of the Tensor will be modified
    /// using the current batch size. The data will be prefilled with zeros.
    /// </summary>
    public class TensorGenerator
    {
        public interface Generator
        {
            /// <summary>
            /// Modifies the data inside a Tensor according to the information contained in the
            /// AgentInfos contained in the current batch.
            /// </summary>
            /// <param name="tensor"> The tensor the data and shape will be modified</param>
            /// <param name="batchSize"> The number of agents present in the current batch</param>
            /// <param name="agentInfo"> Dictionary of Agent to AgentInfo containing the
            /// information that will be used to populate the tensor's data</param>
            void Generate(Tensor tensor, int batchSize, Dictionary<Agent, AgentInfo> agentInfo);
        }
        
        Dictionary<string, Generator> _dict = new Dictionary<string, Generator>();

        /// <summary>
        /// Returns a new TensorGenerators object.
        /// </summary>
        /// <param name="bp"> The BrainParameters used to determine what Generators will be
        /// used</param>
        /// <param name="seed"> The seed the Generators will be initialized with.</param>
        public TensorGenerator(BrainParameters bp, int seed, object barracudaModel = null)
        {
            // Generator for Inputs
            _dict[TensorNames.BatchSizePlaceholder] = new BatchSizeGenerator();
            _dict[TensorNames.SequenceLengthPlaceholder] = new SequenceLengthGenerator();
            _dict[TensorNames.VectorObservationPlacholder] = new VectorObservationGenerator();
            _dict[TensorNames.RecurrentInPlaceholder] = new RecurrentInputGenerator();
            
            #if ENABLE_BARRACUDA
            Barracuda.Model model = (Barracuda.Model) barracudaModel;
            for (var i = 0; i < model?.memories.Length; i++)
            {
                _dict[model.memories[i].input] = new BarracudaRecurrentInputGenerator(i);
            }
            #endif
            
            _dict[TensorNames.PreviousActionPlaceholder] = new PreviousActionInputGenerator();
            _dict[TensorNames.ActionMaskPlaceholder] = new ActionMaskInputGenerator();
            _dict[TensorNames.RandomNormalEpsilonPlaceholder] = new RandomNormalInputGenerator(seed);
            if (bp.cameraResolutions != null)
            {
                for (var visIndex = 0;
                    visIndex < bp.cameraResolutions.Length;
                    visIndex++)
                {
                    var index = visIndex;
                    var bw = bp.cameraResolutions[visIndex].blackAndWhite;
                    _dict[TensorNames.VisualObservationPlaceholderPrefix + visIndex] = new
                            VisualObservationInputGenerator(index, bw);
                }
            }

            // Generators for Outputs
            _dict[TensorNames.ActionOutput] = new BiDimensionalOutputGenerator();
            _dict[TensorNames.RecurrentOutput] = new BiDimensionalOutputGenerator();
            _dict[TensorNames.ValueEstimateOutput] = new BiDimensionalOutputGenerator();
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
        public void GenerateTensors(IEnumerable<Tensor> tensors, 
            int currentBatchSize, 
            Dictionary<Agent, AgentInfo> agentInfos)
        {
            foreach (var tensor in tensors)
            {
                if (!_dict.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as input : " + tensor.Name);
                }
                _dict[tensor.Name].Generate(tensor, currentBatchSize, agentInfos);
            }
        }
    }
}
