using System.Collections.Generic;
using Unity.Sentis;
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
                TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos);
        }

        readonly Dictionary<string, IGenerator> m_Dict = new Dictionary<string, IGenerator>();
        int m_ApiVersion;

        /// <summary>
        /// Returns a new TensorGenerators object.
        /// </summary>
        /// <param name="seed"> The seed the Generators will be initialized with.</param>
        /// <param name="allocator"> Tensor allocator.</param>
        /// <param name="memories">Dictionary of AgentInfo.id to memory for use in the inference model.</param>
        /// <param name="sentisModel"></param>
        /// <param name="deterministicInference"> Inference only: set to true if the action selection from model should be
        /// deterministic. </param>
        public TensorGenerator(
            int seed,
            ITensorAllocator allocator,
            Dictionary<int, List<float>> memories,
            object sentisModel = null,
            bool deterministicInference = false)
        {
            // If model is null, no inference to run and exception is thrown before reaching here.
            if (sentisModel == null)
            {
                return;
            }
            var model = (Model)sentisModel;

            m_ApiVersion = model.GetVersion();

            // Generator for Inputs
            m_Dict[TensorNames.BatchSizePlaceholder] =
                new BatchSizeGenerator(allocator);
            m_Dict[TensorNames.SequenceLengthPlaceholder] =
                new SequenceLengthGenerator(allocator);
            m_Dict[TensorNames.RecurrentInPlaceholder] =
                new RecurrentInputGenerator(allocator, memories);

            m_Dict[TensorNames.PreviousActionPlaceholder] =
                new PreviousActionInputGenerator(allocator);
            m_Dict[TensorNames.ActionMaskPlaceholder] =
                new ActionMaskInputGenerator(allocator);
            m_Dict[TensorNames.RandomNormalEpsilonPlaceholder] =
                new RandomNormalInputGenerator(seed, allocator);


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
        }

        public void InitializeObservations(List<ISensor> sensors, ITensorAllocator allocator)
        {
            if (m_ApiVersion == (int)SentisModelParamLoader.ModelApiVersion.MLAgents1_0)
            {
                // Loop through the sensors on a representative agent.
                // All vector observations use a shared ObservationGenerator since they are concatenated.
                // All other observations use a unique ObservationInputGenerator
                var visIndex = 0;
                ObservationGenerator vecObsGen = null;
                for (var sensorIndex = 0; sensorIndex < sensors.Count; sensorIndex++)
                {
                    var sensor = sensors[sensorIndex];
                    var rank = sensor.GetObservationSpec().Rank;
                    ObservationGenerator obsGen = null;
                    string obsGenName = null;
                    switch (rank)
                    {
                        case 1:
                            if (vecObsGen == null)
                            {
                                vecObsGen = new ObservationGenerator(allocator);
                            }
                            obsGen = vecObsGen;
                            obsGenName = TensorNames.VectorObservationPlaceholder;
                            break;
                        case 2:
                            // If the tensor is of rank 2, we use the index of the sensor
                            // to create the name
                            obsGen = new ObservationGenerator(allocator);
                            obsGenName = TensorNames.GetObservationName(sensorIndex);
                            break;
                        case 3:
                            // If the tensor is of rank 3, we use the "visual observation
                            // index", which only counts the rank 3 sensors
                            obsGen = new ObservationGenerator(allocator);
                            obsGenName = TensorNames.GetVisualObservationName(visIndex);
                            visIndex++;
                            break;
                        default:
                            throw new UnityAgentsException(
                                $"Sensor {sensor.GetName()} have an invalid rank {rank}");
                    }
                    obsGen.AddSensorIndex(sensorIndex);
                    m_Dict[obsGenName] = obsGen;
                }
            }

            if (m_ApiVersion == (int)SentisModelParamLoader.ModelApiVersion.MLAgents2_0)
            {
                for (var sensorIndex = 0; sensorIndex < sensors.Count; sensorIndex++)
                {
                    var obsGen = new ObservationGenerator(allocator);
                    var obsGenName = TensorNames.GetObservationName(sensorIndex);
                    obsGen.AddSensorIndex(sensorIndex);
                    m_Dict[obsGenName] = obsGen;
                }
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
            IReadOnlyList<TensorProxy> tensors, int currentBatchSize, IList<AgentInfoSensorsPair> infos)
        {
            for (var tensorIndex = 0; tensorIndex < tensors.Count; tensorIndex++)
            {
                var tensor = tensors[tensorIndex];
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
