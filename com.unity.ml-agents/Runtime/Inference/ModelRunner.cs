using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TransformsAI.MicroMLAgents.Actuators;
using TransformsAI.MicroMLAgents.Sensors;
using Unity.Barracuda;
using UnityEngine.Profiling;

namespace TransformsAI.MicroMLAgents.Inference
{
    public class ModelRunner : IDisposable
    {
        private static readonly ThreadLocal<Stack<List<IAgent>>> Pool = new(() => new Stack<List<IAgent>>());

        ITensorAllocator m_TensorAllocator;
        TensorGenerator m_TensorGenerator;
        TensorApplier m_TensorApplier;

        IWorker m_Engine;
        bool m_Verbose = false;
        bool m_DeterministicInference;
        string[] m_OutputNames;
        IReadOnlyList<TensorProxy> m_InferenceInputs;
        List<TensorProxy> m_InferenceOutputs;
        Dictionary<string, Tensor> m_InputsByName;
        private readonly int m_Seed;
        public Model Model { get; }
        private ObservationSpec[] m_ObservationSpecs;
        private ActionSpec m_ActionSpec;
        private readonly string m_Name;

        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="model"> The Barracuda model to load </param>
        /// <param name="actionSpec"> Description of the actions for the Agent.</param>
        /// <param name="verbose"> If true, uses verbose .</param>
        /// <param name="inferenceDevice"> Inference execution device. CPU is the fastest
        /// option for most of ML Agents models. </param>
        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
        /// and Multinomial objects used when running inference.</param>
        /// <param name="deterministicInference"> Inference only: set to true if the action selection from model should be
        /// deterministic. </param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
        public ModelRunner(
            string name,
            Model model,
            WorkerFactory.Device inferenceDevice = WorkerFactory.Device.Auto,
            int seed = 0,
            bool deterministicInference = false,
            bool verbose = false)
        {
            m_DeterministicInference = deterministicInference;
            m_TensorAllocator = new TensorCachingAllocator();

            m_Verbose = verbose;
            D.logEnabled = m_Verbose;

            const int mlAgents20Version = 3;
            if (model.GetVersion() != mlAgents20Version)
                throw new Exception("Invalid Version");

            Model = model;
            m_Engine = WorkerFactory.CreateWorker(model, inferenceDevice, m_Verbose);

            m_InferenceInputs = model.GetInputTensors();
            m_OutputNames = model.GetOutputNames(m_DeterministicInference);
            m_Seed = seed;
            m_InputsByName = new Dictionary<string, Tensor>();
            m_InferenceOutputs = new List<TensorProxy>();
            m_Name = name;
        }


        public void Dispose()
        {
            if (m_Engine != null) m_Engine.Dispose();
            m_TensorAllocator?.Reset(false);
            foreach (var tensor in m_InputsByName.Values) tensor.Dispose();
            foreach (var tensor in m_InferenceOutputs) tensor.data.Dispose();
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


        void PrepareBarracudaInputs(IReadOnlyList<TensorProxy> infInputs)
        {
            m_InputsByName.Clear();
            for (var i = 0; i < infInputs.Count; i++)
            {
                var inp = infInputs[i];
                m_InputsByName[inp.name] = inp.data;
            }
        }

        public void Initialize(IList<ObservationSpec> observationSpecs, ActionSpec actionSpec)
        {
            if (m_TensorGenerator != null || m_TensorApplier != null)
                throw new Exception("Model Runner Already Initialized");

            m_ObservationSpecs = observationSpecs.ToArray();
            m_ActionSpec = actionSpec;
            m_TensorGenerator = new TensorGenerator(m_Seed, observationSpecs, m_TensorAllocator, Model,
                m_DeterministicInference);
            m_TensorApplier = new TensorApplier(m_Seed, actionSpec, m_TensorAllocator, Model, m_DeterministicInference);
        }

        public void DecideBatch(List<IAgent> m_Infos)
        {
            var currentBatchSize = m_Infos.Count;
            if (currentBatchSize == 0)
            {
                return;
            }


            foreach (var agent in m_Infos)
            {
                if (agent.SensorObservationSpecs.Length != m_Infos.Count) throw new Exception("Sensor Count Mismatch");
                for (var i = 0; i < agent.SensorObservationSpecs.Length; i++)
                {
                    var agentSpec = agent.SensorObservationSpecs[i];
                    var modelSpec = m_ObservationSpecs[i];

                    if (agentSpec != modelSpec) throw new Exception($"Observation mismatch on sensor {i}");
                }

                if (agent.ActionSpec != m_ActionSpec) throw new Exception("Action spec mismatch");
            }
            // Todo(Dante): check that all agents have the same ObsSpecs and ActionSpec

            Profiler.BeginSample("ModelRunner.DecideAction");
            Profiler.BeginSample(m_Name);

            Profiler.BeginSample($"GenerateTensors");
            // Prepare the input tensors to be feed into the engine
            m_TensorGenerator.GenerateTensors(m_InferenceInputs, m_Infos);
            Profiler.EndSample();

            Profiler.BeginSample($"PrepareBarracudaInputs");
            PrepareBarracudaInputs(m_InferenceInputs);
            Profiler.EndSample();

            // Execute the Model
            Profiler.BeginSample($"ExecuteGraph");
            m_Engine.Execute(m_InputsByName);
            Profiler.EndSample();

            Profiler.BeginSample($"FetchBarracudaOutputs");
            FetchBarracudaOutputs(m_OutputNames);
            Profiler.EndSample();

            Profiler.BeginSample($"ApplyTensors");
            // Update the outputs
            m_TensorApplier.ApplyTensors(m_InferenceOutputs, m_Infos);
            Profiler.EndSample();

            Profiler.EndSample(); // end name
            Profiler.EndSample(); // end ModelRunner.DecideAction
        }

        public void Decide(IAgent agent)
        {
            var agentList = Pool.Value.Count > 0 ? Pool.Value.Pop() : new List<IAgent>();
            try
            {
                agentList.Add(agent);

                DecideBatch(agentList);
            }
            finally
            {
                Pool.Value.Push(agentList);
            }
        }
    }
}
