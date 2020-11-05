using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Inference
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
            void Generate(
                TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos);
        }

        readonly Dictionary<string, IGenerator> m_Dict = new Dictionary<string, IGenerator>();

        /// <summary>
        /// Returns a new TensorGenerators object.
        /// </summary>
        /// <param name="seed"> The seed the Generators will be initialized with.</param>
        /// <param name="allocator"> Tensor allocator.</param>
        /// <param name="memories">Dictionary of AgentInfo.id to memory for use in the inference model.</param>
        /// <param name="barracudaModel"></param>
        public TensorGenerator(
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

            // Generator for Inputs
            m_Dict[TensorNames.BatchSizePlaceholder] =
                new BatchSizeGenerator(allocator);
            m_Dict[TensorNames.SequenceLengthPlaceholder] =
                new SequenceLengthGenerator(allocator);
            m_Dict[TensorNames.RecurrentInPlaceholder] =
                new RecurrentInputGenerator(allocator, memories);

            for (var i = 0; i < model.memories.Count; i++)
            {
                m_Dict[model.memories[i].input] =
                    new BarracudaRecurrentInputGenerator(i, allocator, memories);
            }

            m_Dict[TensorNames.PreviousActionPlaceholder] =
                new PreviousActionInputGenerator(allocator);
            m_Dict[TensorNames.ActionMaskPlaceholder] =
                new ActionMaskInputGenerator(allocator);
            m_Dict[TensorNames.RandomNormalEpsilonPlaceholder] =
                new RandomNormalInputGenerator(seed, allocator);


            // Generators for Outputs
            if (model.HasContinuousOutputs())
            {
                m_Dict[model.ContinuousOutputName()] = new BiDimensionalOutputGenerator(allocator);
            }
            if (model.HasDiscreteOutputs())
            {
                m_Dict[model.DiscreteOutputName()] = new BiDimensionalOutputGenerator(allocator);
            }
            m_Dict[TensorNames.RecurrentOutput] = new BiDimensionalOutputGenerator(allocator);
            m_Dict[TensorNames.ValueEstimateOutput] = new BiDimensionalOutputGenerator(allocator);
        }

        public void InitializeObservations(List<ISensor> sensors, ITensorAllocator allocator)
        {
            // Loop through the sensors on a representative agent.
            // For vector observations, add the index to the (single) VectorObservationGenerator
            // For visual observations, make a VisualObservationInputGenerator
            var visIndex = 0;
            VectorObservationGenerator vecObsGen = null;
            for (var sensorIndex = 0; sensorIndex < sensors.Count; sensorIndex++)
            {
                var sensor = sensors[sensorIndex];
                var shape = sensor.GetObservationShape();
                // TODO generalize - we currently only have vector or visual, but can't handle "2D" observations
                var isVectorSensor = (shape.Length == 1);
                if (isVectorSensor)
                {
                    if (vecObsGen == null)
                    {
                        vecObsGen = new VectorObservationGenerator(allocator);
                    }

                    vecObsGen.AddSensorIndex(sensorIndex);
                }
                else
                {
                    m_Dict[TensorNames.VisualObservationPlaceholderPrefix + visIndex] =
                        new VisualObservationInputGenerator(sensorIndex, allocator);
                    visIndex++;
                }
            }

            if (vecObsGen != null)
            {
                m_Dict[TensorNames.VectorObservationPlaceholder] = vecObsGen;
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
        public void GenerateTensors(
            IEnumerable<TensorProxy> tensors, int currentBatchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            foreach (var tensor in tensors)
            {
                if (!m_Dict.ContainsKey(tensor.name))
                {
                    throw new UnityAgentsException(
                        $"Unknown tensorProxy expected as input : {tensor.name}");
                }
                m_Dict[tensor.name].Generate(tensor, currentBatchSize, infos);
            }
        }
    }
}
