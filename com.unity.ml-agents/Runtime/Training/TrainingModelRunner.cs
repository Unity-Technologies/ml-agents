// ModelRunner for C# training.

using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents.Inference.Utils;
using System.Linq;

namespace Unity.MLAgents
{
    internal class TrainingModelRunner
    {
        List<AgentInfoSensorsPair> m_Infos = new List<AgentInfoSensorsPair>();
        Dictionary<int, ActionBuffers> m_LastActionsReceived = new Dictionary<int, ActionBuffers>();
        List<int> m_OrderedAgentsRequestingDecisions = new List<int>();
        TensorProxy m_TrainingState;

        ITensorAllocator m_TensorAllocator;
        TensorGenerator m_TensorGenerator;
        TrainingTensorGenerator m_TrainingTensorGenerator;
        TensorApplier m_TensorApplier;

        Model m_Model;
        IWorker m_Engine;
        bool m_Verbose = false;
        string[] m_TrainingOutputNames;
        IReadOnlyList<TensorProxy> m_TrainingInputs;
        IReadOnlyList<TensorProxy> m_InferenceInputs;
        List<TensorProxy> m_TrainingOutputs;
        Dictionary<string, Tensor> m_InputsByName;
        Dictionary<int, List<float>> m_Memories = new Dictionary<int, List<float>>();

        bool m_ObservationsInitialized;
        bool m_TrainingObservationsInitialized;

        ReplayBuffer m_Buffer;

        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="model"> The Barracuda model to load </param>
        /// <param name="actionSpec"> Description of the actions for the Agent.</param>
        /// <param name="inferenceDevice"> Inference execution device. CPU is the fastest
        /// option for most of ML Agents models. </param>
        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
        /// and Multinomial objects used when running inference.</param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
        public TrainingModelRunner(
            ActionSpec actionSpec,
            NNModel model,
            ReplayBuffer buffer,
            TrainerConfig config,
            int seed = 0)
        {
            Model barracudaModel;
            m_TensorAllocator = new TensorCachingAllocator();

            // barracudaModel = Barracuda.SomeModelBuilder.CreateModel();
            barracudaModel = ModelLoader.Load(model);
            m_Model = barracudaModel;
            WorkerFactory.Type executionDevice = WorkerFactory.Type.CSharpBurst;
            m_Engine = WorkerFactory.CreateWorker(executionDevice, barracudaModel, m_Verbose);

            m_TrainingInputs = barracudaModel.GetTrainingInputTensors();
            List<TensorProxy> infTensors = new List<TensorProxy>();
            foreach(var tensor in m_TrainingInputs)
            {
                if (tensor.name == TensorNames.Observations || tensor.name == TensorNames.BatchSizePlaceholder)
                {
                    infTensors.Add(tensor);
                }
            }
            m_InferenceInputs = (IReadOnlyList<TensorProxy>) infTensors;
            m_TrainingOutputNames = barracudaModel.GetTrainingOutputNames();
            m_TensorGenerator = new TensorGenerator(
                seed, m_TensorAllocator, m_Memories, barracudaModel);
            m_TrainingTensorGenerator = new TrainingTensorGenerator(
                seed, m_TensorAllocator, config.learningRate, config.gamma, barracudaModel);
            m_TensorApplier = new TensorApplier(
                actionSpec, seed, m_TensorAllocator, m_Memories, barracudaModel);
            m_InputsByName = new Dictionary<string, Tensor>();
            m_TrainingOutputs = new List<TensorProxy>();
            m_Buffer = buffer;
            InitializeTrainingState();
        }

        void InitializeTrainingState()
        {
            var initState = m_Model.GetTensorByName(TensorNames.InitialTrainingState);
            m_TrainingState = new TensorProxy{
                name = TensorNames.InitialTrainingState,
                valueType = TensorProxy.TensorType.FloatingPoint,
                data = initState,
                shape = initState.shape.ToArray().Select(i => (long)i).ToArray()
            };
        }

        void PrepareBarracudaInputs(IReadOnlyList<TensorProxy> infInputs)
        {
            m_InputsByName.Clear();
            for (var i = 0; i < infInputs.Count; i++)
            {
                var inp = infInputs[i];
                m_InputsByName[inp.name] = inp.data;
            }
        }

        public ITensorAllocator Allocator
        {
            get => m_TensorAllocator;
        }

        public void Dispose()
        {
            if (m_Engine != null)
                m_Engine.Dispose();
            m_TensorAllocator?.Reset(false);
        }

        void FetchBarracudaOutputs(string[] names)
        {
            m_TrainingOutputs.Clear();
            foreach (var n in names)
            {
                var output = m_Engine.PeekOutput(n);
                m_TrainingOutputs.Add(TensorUtils.TensorProxyFromBarracuda(output, n));
            }
        }

        public void PutObservations(AgentInfo info, List<ISensor> sensors)
        {
            m_Infos.Add(new AgentInfoSensorsPair
            {
                agentInfo = info,
                sensors = sensors
            });

            // We add the episodeId to this list to maintain the order in which the decisions were requested
            m_OrderedAgentsRequestingDecisions.Add(info.episodeId);

            if (!m_LastActionsReceived.ContainsKey(info.episodeId))
            {
                m_LastActionsReceived[info.episodeId] = ActionBuffers.Empty;
            }
            if (info.done)
            {
                // If the agent is done, we remove the key from the last action dictionary since no action
                // should be taken.
                m_LastActionsReceived.Remove(info.episodeId);
            }
        }

        public void GetObservationTensors(IReadOnlyList<TensorProxy> tensors, AgentInfo info, List<ISensor> sensors)
        {
            if (!m_ObservationsInitialized)
            {
                // Just grab the first agent in the collection (any will suffice, really).
                // We check for an empty Collection above, so this will always return successfully.
                m_TensorGenerator.InitializeObservations(sensors, m_TensorAllocator);
                m_ObservationsInitialized = true;
            }
            var infoSensorPair = new AgentInfoSensorsPair
            {
                agentInfo = info,
                sensors = sensors
            };
            m_TensorGenerator.GenerateTensors(tensors, 1, new List<AgentInfoSensorsPair> { infoSensorPair });
        }

        public IReadOnlyList<TensorProxy> GetInputTensors()
        {
            return m_Model.GetTrainingObservationInputTensors();
        }

        public void DecideBatch()
        {
            var currentBatchSize = m_Infos.Count;
            if (currentBatchSize == 0)
            {
                return;
            }
            if (!m_ObservationsInitialized)
            {
                // Just grab the first agent in the collection (any will suffice, really).
                // We check for an empty Collection above, so this will always return successfully.
                var firstInfo = m_Infos[0];
                m_TensorGenerator.InitializeObservations(firstInfo.sensors, m_TensorAllocator);
                m_ObservationsInitialized = true;
            }

            // Prepare the input tensors to be feed into the engine
            m_TensorGenerator.GenerateTensors(m_InferenceInputs, currentBatchSize, m_Infos);
            m_TrainingTensorGenerator.GenerateTensors(m_TrainingInputs, currentBatchSize, m_Buffer.SampleDummyBatch(currentBatchSize), m_TrainingState);

            PrepareBarracudaInputs(m_TrainingInputs);

            // Execute the Model
            m_Engine.Execute(m_InputsByName);

            FetchBarracudaOutputs(m_TrainingOutputNames);

            // Update the outputs
            m_TensorApplier.ApplyTensors(m_TrainingOutputs, m_OrderedAgentsRequestingDecisions, m_LastActionsReceived);

            m_Infos.Clear();

            m_OrderedAgentsRequestingDecisions.Clear();
        }

        public void UpdateModel(List<Transition> transitions)
        {
            var currentBatchSize = transitions.Count;
            if (currentBatchSize == 0)
            {
                return;
            }

            m_TrainingTensorGenerator.GenerateTensors(m_TrainingInputs, currentBatchSize, transitions, m_TrainingState, true);

            PrepareBarracudaInputs(m_TrainingInputs);

            // Execute the Model
            m_Engine.Execute(m_InputsByName);

            FetchBarracudaOutputs(m_TrainingOutputNames);

            // Update the model
            // m_TensorApplier.UpdateModel(m_TrainingOutputs, m_OrderedAgentsRequestingDecisions, m_LastActionsReceived);
        }

        public ActionBuffers GetAction(int agentId)
        {
            if (m_LastActionsReceived.ContainsKey(agentId))
            {
                return m_LastActionsReceived[agentId];
            }
            return ActionBuffers.Empty;
        }

        // void PrintTensor(TensorProxy tensor)
        // {
        //     Debug.Log($"Print tensor {tensor.name}");
        //     for (var b = 0; b < tensor.data.batch; b++)
        //     {
        //         var message = new List<float>();
        //         for (var i = 0; i < tensor.data.height; i++)
        //         {
        //             for (var j = 0; j < tensor.data.width; j++)
        //             {
        //                 for(var k = 0; k < tensor.data.channels; k++)
        //                 {
        //                     message.Add(tensor.data[b, i, j, k]);
        //                 }
        //             }
        //         }
        //         Debug.Log(string.Join(", ", message));
        //     }
        // }
    }
}
