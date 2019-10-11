using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using UnityEngine.Profiling;

namespace MLAgents.InferenceBrain
{
    public class BrainModelRunner : IBatchedDecisionMaker
    {
        private ITensorAllocator m_TensorAllocator;
        private TensorGenerator m_TensorGenerator;
        private TensorApplier m_TensorApplier;

        private NNModel m_Model;
        private Model m_BarracudaModel;
        private IWorker m_Engine;
        private bool m_Verbose = false;
        private string[] m_OutputNames;
        private IReadOnlyList<TensorProxy> m_InferenceInputs;
        private IReadOnlyList<TensorProxy> m_InferenceOutputs;

        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="model"> The Barracuda model to load
        /// <param name="brainParameters"> The parameters of the Brain used to generate the 
        /// placeholder tensors
        /// <param name="inferenceDevice"> Inference execution device. CPU is the fastest 
        /// option for most of ML Agents models.
        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
        /// and Multinomial objects used when running inference.</param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
        public BrainModelRunner(
            NNModel model,
            BrainParameters brainParameters,
            InferenceDevice inferenceDevice = InferenceDevice.CPU,
            int seed = 0)
        {
            m_Model = model;
            m_TensorAllocator = new TensorCachingAllocator();
            if (model != null)
            {
#if BARRACUDA_VERBOSE
                _verbose = true;
#endif

                D.logEnabled = m_Verbose;

                // Cleanup previous instance
                m_Engine.Dispose();

                m_BarracudaModel = ModelLoader.Load(model.Value);
                var executionDevice = inferenceDevice == InferenceDevice.GPU
                    ? BarracudaWorkerFactory.Type.ComputePrecompiled
                    : BarracudaWorkerFactory.Type.CSharp;
                m_Engine = BarracudaWorkerFactory.CreateWorker(executionDevice, m_BarracudaModel, m_Verbose);
            }
            else
            {
                m_BarracudaModel = null;
                m_Engine = null;
            }

            m_InferenceInputs = BarracudaModelParamLoader.GetInputTensors(m_BarracudaModel);
            m_OutputNames = BarracudaModelParamLoader.GetOutputNames(m_BarracudaModel);
            m_TensorGenerator = new TensorGenerator(brainParameters, seed, m_TensorAllocator, m_BarracudaModel);
            m_TensorApplier = new TensorApplier(brainParameters, seed, m_TensorAllocator, m_BarracudaModel);
        }

        private Dictionary<string, Tensor> PrepareBarracudaInputs(IEnumerable<TensorProxy> infInputs)
        {
            var inputs = new Dictionary<string, Tensor>();
            foreach (var inp in m_InferenceInputs)
            {
                inputs[inp.name] = inp.data;
            }

            return inputs;
        }
        public void Dispose()
        {
            if (m_Engine != null)
                m_Engine.Dispose();
            m_TensorAllocator?.Reset(false);
        }

        private List<TensorProxy> FetchBarracudaOutputs(string[] names)
        {
            var outputs = new List<TensorProxy>();
            foreach (var n in names)
            {
                var output = m_Engine.Peek(n);
                outputs.Add(TensorUtils.TensorProxyFromBarracuda(output, n));
            }

            return outputs;
        }

        public void PutObservations(string key, ICollection<Agent> agents)
        {
            var currentBatchSize = agents.Count;
            if (currentBatchSize == 0)
            {
                return;
            }

            Profiler.BeginSample("LearningBrain.DecideAction");
            if (m_Engine == null)
            {
                Debug.LogError($"No model was present for the Brain {m_Model.name}.");
                return;
            }

            Profiler.BeginSample($"MLAgents.{m_Model.name}.GenerateTensors");
            // Prepare the input tensors to be feed into the engine
            m_TensorGenerator.GenerateTensors(m_InferenceInputs, currentBatchSize, agents);
            Profiler.EndSample();

            Profiler.BeginSample($"MLAgents.{m_Model.name}.PrepareBarracudaInputs");
            var inputs = PrepareBarracudaInputs(m_InferenceInputs);
            Profiler.EndSample();

            // Execute the Model
            Profiler.BeginSample($"MLAgents.{m_Model.name}.ExecuteGraph");
            m_Engine.Execute(inputs);
            Profiler.EndSample();

            Profiler.BeginSample($"MLAgents.{m_Model.name}.FetchBarracudaOutputs");
            m_InferenceOutputs = FetchBarracudaOutputs(m_OutputNames);
            Profiler.EndSample();

            Profiler.BeginSample($"MLAgents.{m_Model.name}.ApplyTensors");
            // Update the outputs
            m_TensorApplier.ApplyTensors(m_InferenceOutputs, agents);
            Profiler.EndSample();

            Profiler.EndSample();
        }

        public bool HasModel(NNModel other)
        {
            return m_Model == other;
        }
    }
}
