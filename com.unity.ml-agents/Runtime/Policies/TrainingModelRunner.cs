// ModelRunner for C# training.

using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine.Profiling;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents
{
    internal class TrainingModelRunner
    {
        List<AgentInfoSensorsPair> m_Infos = new List<AgentInfoSensorsPair>();
        Dictionary<int, ActionBuffers> m_LastActionsReceived = new Dictionary<int, ActionBuffers>();
        List<int> m_OrderedAgentsRequestingDecisions = new List<int>();

        ITensorAllocator m_TensorAllocator;
        TensorGenerator m_TensorGenerator;
        TensorApplier m_TensorApplier;

        Model m_Model;
        NNModel m_TargetModel;
        string m_ModelName;
        InferenceDevice m_InferenceDevice;
        IWorker m_Engine;
        bool m_Verbose = false;
        string[] m_OutputNames;
        IReadOnlyList<TensorProxy> m_InferenceInputs;
        List<TensorProxy> m_InferenceOutputs;
        Dictionary<string, Tensor> m_InputsByName;
        Dictionary<int, List<float>> m_Memories = new Dictionary<int, List<float>>();

        SensorShapeValidator m_SensorShapeValidator = new SensorShapeValidator();

        bool m_ObservationsInitialized;

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
            int seed = 0)
        {
            Model barracudaModel;
            m_TensorAllocator = new TensorCachingAllocator();

            // barracudaModel = Barracuda.SomeModelBuilder.CreateModel();
            // barracudaModel = ModelLoader.Load(new NNModel());
            barracudaModel = ModelLoader.Load(model);
            m_Model = barracudaModel;
            WorkerFactory.Type executionDevice = WorkerFactory.Type.CSharp;
            m_Engine = WorkerFactory.CreateWorker(executionDevice, barracudaModel, m_Verbose);

            m_InferenceInputs = barracudaModel.GetInputTensors();
            m_OutputNames = barracudaModel.GetOutputNames();
            m_TensorGenerator = new TensorGenerator(
                seed, m_TensorAllocator, m_Memories, barracudaModel);
            m_TensorApplier = new TensorApplier(
                actionSpec, seed, m_TensorAllocator, m_Memories, barracudaModel);
            m_InputsByName = new Dictionary<string, Tensor>();
            m_InferenceOutputs = new List<TensorProxy>();
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

        public void Dispose()
        {
            if (m_Engine != null)
                m_Engine.Dispose();
            m_TensorAllocator?.Reset(false);
        }

        void FetchBarracudaOutputs(string[] names)
        {
            m_InferenceOutputs.Clear();
            foreach (var n in names)
            {
                var output = m_Engine.PeekOutput(n);
                m_InferenceOutputs.Add(TensorUtils.TensorProxyFromBarracuda(output, n));
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
            return m_Model.GetInputTensors();
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

            PrepareBarracudaInputs(m_InferenceInputs);

            // Execute the Model
            m_Engine.Execute(m_InputsByName);

            FetchBarracudaOutputs(m_OutputNames);

            // Update the outputs
            m_TensorApplier.ApplyTensors(m_InferenceOutputs, m_OrderedAgentsRequestingDecisions, m_LastActionsReceived);

            m_Infos.Clear();

            m_OrderedAgentsRequestingDecisions.Clear();
        }

        public ActionBuffers GetAction(int agentId)
        {
            if (m_LastActionsReceived.ContainsKey(agentId))
            {
                return m_LastActionsReceived[agentId];
            }
            return ActionBuffers.Empty;
        }
    }
}
