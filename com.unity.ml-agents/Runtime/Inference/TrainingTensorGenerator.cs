using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Sensors;
using Unity.MLAgents;
using UnityEngine;

namespace Unity.MLAgents.Inference
{

    internal class TrainingTensorGenerator
    {
        public interface ITrainingGenerator
        {
            void Generate(
                TensorProxy tensorProxy, int batchSize, IList<Transition> transitions);
        }

        readonly Dictionary<string, ITrainingGenerator> m_Dict = new Dictionary<string, ITrainingGenerator>();


        public TrainingTensorGenerator(
            int seed,
            ITensorAllocator allocator,
            object barracudaModel = null)
        {
            // If model is null, no inference to run and exception is thrown before reaching here.
            if (barracudaModel == null)
            {
                return;
            }
            var model = (Model)barracudaModel;

            // Generator for Inputs
            m_Dict[TensorNames.ActionInput] = new ActionInputGenerator(allocator);
            m_Dict[TensorNames.TargetInput] = new RewardInputGenerator(allocator);
            m_Dict[TensorNames.RewardInput] = new RewardInputGenerator(allocator);
            m_Dict[TensorNames.LearningRate] = new ConstantGenerator(allocator, 0.0001f);
            m_Dict[TensorNames.BatchSizePlaceholder] = new TrainingBatchSizeGenerator(allocator);

            // Generators for Outputs
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
            IReadOnlyList<TensorProxy> tensors, int currentBatchSize, IList<Transition> transitions, bool training=false)
        {
            for (var tensorIndex = 0; tensorIndex < tensors.Count; tensorIndex++)
            {
                var tensor = tensors[tensorIndex];
                if (!m_Dict.ContainsKey(tensor.name))
                {
                    throw new UnityAgentsException(
                        $"Unknown tensorProxy expected as input : {tensor.name}");
                }
                if (tensor.name.StartsWith("obs_") || tensor.name == TensorNames.BatchSizePlaceholder)
                {
                    if (training == true)
                    {
                        m_Dict[tensor.name].Generate(tensor, currentBatchSize, transitions);
                    }
                }
                else
                {
                    m_Dict[tensor.name].Generate(tensor, currentBatchSize, transitions);
                }
            }
        }

        public void InitializeObservations(Transition transition, ITensorAllocator allocator)
        {
            for (var sensorIndex = 0; sensorIndex < transition.state.Count; sensorIndex++)
            {
                var obsGen = new CopyObservationTensorsGenerator(allocator);
                var obsGenName = TensorNames.GetObservationName(sensorIndex);
                obsGen.SetSensorIndex(sensorIndex);
                m_Dict[obsGenName] = obsGen;
            }

            for (var sensorIndex = 0; sensorIndex < transition.nextState.Count; sensorIndex++)
            {
                var obsGen = new CopyNextObservationTensorsGenerator(allocator);
                var obsGenName = TensorNames.GetNextObservationName(sensorIndex);
                obsGen.SetSensorIndex(sensorIndex);
                m_Dict[obsGenName] = obsGen;
            }
        }

        public static void CopyTensorToBatch(TensorProxy source, TensorProxy target, int batchIndex)
        {
            for (var i = 0; i < source.Height; i++)
            {
                for (var j = 0; j < source.Width; j++)
                {
                    for(var k = 0; k < source.Channels; k++)
                    {
                        target.data[batchIndex, i, j, k] = source.data[0, i, j, k];
                    }
                }
            }
        }
    }

    internal class ActionInputGenerator: TrainingTensorGenerator.ITrainingGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public ActionInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<Transition> transitions)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            for (var index = 0; index < batchSize; index++)
            {
                var actions = transitions[index].action.DiscreteActions;
                for (var j = 0; j < actions.Length; j++)
                {
                    tensorProxy.data[index, j] = actions[j];
                }
            }
        }
    }

    internal class RewardInputGenerator: TrainingTensorGenerator.ITrainingGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public RewardInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<Transition> transitions)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            for (var index = 0; index < batchSize; index++)
            {
                tensorProxy.data[index, 0] = transitions[index].reward;
            }
        }
    }

    internal class CopyObservationTensorsGenerator: TrainingTensorGenerator.ITrainingGenerator
    {
        readonly ITensorAllocator m_Allocator;

        int m_SensorIndex;

        public CopyObservationTensorsGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void SetSensorIndex(int index)
        {
            m_SensorIndex = index;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<Transition> transitions)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            for (var index = 0; index < batchSize; index++)
            {
                TrainingTensorGenerator.CopyTensorToBatch(transitions[index].state[m_SensorIndex], tensorProxy, index);
            }
        }
    }

    internal class CopyNextObservationTensorsGenerator: TrainingTensorGenerator.ITrainingGenerator
    {
        readonly ITensorAllocator m_Allocator;

        int m_SensorIndex;

        public CopyNextObservationTensorsGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void SetSensorIndex(int index)
        {
            m_SensorIndex = index;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<Transition> transitions)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            for (var index = 0; index < batchSize; index++)
            {
                TrainingTensorGenerator.CopyTensorToBatch(transitions[index].nextState[m_SensorIndex], tensorProxy, index);
            }
        }
    }

    internal class ConstantGenerator: TrainingTensorGenerator.ITrainingGenerator
    {
        readonly ITensorAllocator m_Allocator;
        float m_Const;

        public ConstantGenerator(ITensorAllocator allocator, float c)
        {
            m_Allocator = allocator;
            m_Const = c;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<Transition> transitions)
        {
            TensorUtils.ResizeTensor(tensorProxy, 1, m_Allocator);
            for (var index = 0; index < batchSize; index++)
            {
                tensorProxy.data?.Dispose();
                tensorProxy.data = m_Allocator.Alloc(new TensorShape(1, 1));
                tensorProxy.data[0] = m_Const;
            }
        }
    }
    internal class TrainingBatchSizeGenerator : TrainingTensorGenerator.ITrainingGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public TrainingBatchSizeGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<Transition> transitions)
        {
            tensorProxy.data?.Dispose();
            tensorProxy.data = m_Allocator.Alloc(new TensorShape(1, 1));
            tensorProxy.data[0] = batchSize;
        }
    }

}
