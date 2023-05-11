using System;
using System.Collections.Generic;
using TransformsAI.MicroMLAgents.Inference.Utils;
using TransformsAI.MicroMLAgents.Sensors;
using Unity.Barracuda;

namespace TransformsAI.MicroMLAgents.Inference
{
    /// <summary>
    /// Reshapes a Tensor so that its first dimension becomes equal to the current batch size
    /// and initializes its content to be zeros. Will only work on 2-dimensional tensors.
    /// The second dimension of the Tensor will not be modified.
    /// </summary>
    internal class BiDimensionalOutputGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public BiDimensionalOutputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            TensorUtils.ResizeTensor(tensorProxy, batch.Count, m_Allocator);
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the BatchSize input : Will be a one dimensional
    /// integer array of size 1 containing the batch size.
    /// </summary>
    internal class BatchSizeGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public BatchSizeGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            tensorProxy.data?.Dispose();
            tensorProxy.data = m_Allocator.Alloc(new TensorShape(1, 1));
            tensorProxy.data[0] = batch.Count;
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the SequenceLength input : Will be a one
    /// dimensional integer array of size 1 containing 1.
    /// Note : the sequence length is always one since recurrent networks only predict for
    /// one step at the time.
    /// </summary>
    internal class SequenceLengthGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public SequenceLengthGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            tensorProxy.shape = Array.Empty<long>();
            tensorProxy.data?.Dispose();
            tensorProxy.data = m_Allocator.Alloc(new TensorShape(1, 1));
            tensorProxy.data[0] = 1;
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Recurrent input : Will be a two
    /// dimensional float array of dimension [batchSize x memorySize].
    /// It will use the Memory data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    internal class RecurrentInputGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public RecurrentInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            TensorUtils.ResizeTensor(tensorProxy, batch.Count, m_Allocator);

            var memorySize = tensorProxy.data.width;
            
            for (var agentIndex = 0; agentIndex < batch.Count; agentIndex++)
            {
                var info = batch[agentIndex];
                var memory = info.Memory;
                
                if (memory == null)
                {
                    for (var j = 0; j < memorySize; j++)
                    {
                        tensorProxy.data[agentIndex, 0, j, 0] = 0;
                    }
                    agentIndex++;
                    continue;
                }
                for (var j = 0; j < Math.Min(memorySize, memory.Count); j++)
                {
                    if (j >= memory.Count)
                    {
                        break;
                    }
                    tensorProxy.data[agentIndex, 0, j, 0] = memory[j];
                }
            }
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Previous Action input : Will be a two
    /// dimensional integer array of dimension [batchSize x actionSize].
    /// It will use the previous action data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    internal class PreviousActionInputGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public PreviousActionInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            TensorUtils.ResizeTensor(tensorProxy, batch.Count, m_Allocator);

            var actionSize = tensorProxy.shape[^1];
            for (var agentIndex = 0; agentIndex < batch.Count; agentIndex++)
            {
                var info = batch[agentIndex];
                var pastAction = info.ActionBuffer.DiscreteActions;
                if (!pastAction.IsEmpty())
                {
                    for (var j = 0; j < actionSize; j++)
                    {
                        tensorProxy.data[agentIndex, j] = pastAction[j];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Action Mask input : Will be a two
    /// dimensional float array of dimension [batchSize x numActionLogits].
    /// It will use the Action Mask data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    internal class ActionMaskInputGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public ActionMaskInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            TensorUtils.ResizeTensor(tensorProxy, batch.Count, m_Allocator);

            var maskSize = tensorProxy.shape[^1];
            for (var agentIndex = 0; agentIndex < batch.Count; agentIndex++)
            {
                var agentInfo = batch[agentIndex];
                var maskList = agentInfo.DiscreteActionMasks;
                for (var j = 0; j < maskSize; j++)
                {
                    var isUnmasked = maskList != null && maskList[j] ? 0.0f : 1.0f;
                    tensorProxy.data[agentIndex, j] = isUnmasked;
                }
            }
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Epsilon input : Will be a two
    /// dimensional float array of dimension [batchSize x actionSize].
    /// It will use the generate random input data from a normal Distribution.
    /// </summary>
    internal class RandomNormalInputGenerator : TensorGenerator.IGenerator
    {
        readonly RandomNormal m_RandomNormal;
        readonly ITensorAllocator m_Allocator;

        public RandomNormalInputGenerator(int seed, ITensorAllocator allocator)
        {
            m_RandomNormal = new RandomNormal(seed);
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            TensorUtils.ResizeTensor(tensorProxy, batch.Count, m_Allocator);
            TensorUtils.FillTensorWithRandomNormal(tensorProxy, m_RandomNormal);
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Observation input : Will be a multi
    /// dimensional float array.
    /// It will use the Observation data contained in the sensors to fill the data
    /// of the tensor.
    /// </summary>
    internal class ObservationGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;
        readonly int m_Index;
        ObservationWriter m_ObservationWriter = new ObservationWriter();

        public ObservationGenerator(ITensorAllocator allocator, int index)
        {
            m_Allocator = allocator;
            m_Index = index;
        }
        
        public void Generate(TensorProxy tensorProxy, IList<IAgent> batch)
        {
            TensorUtils.ResizeTensor(tensorProxy, batch.Count, m_Allocator);
            for (var agentIndex = 0; agentIndex < batch.Count; agentIndex++)
            {
                var info = batch[agentIndex];
                var tensorOffset = 0;

                m_ObservationWriter.SetTarget(tensorProxy, agentIndex, tensorOffset);
                var numWritten = info.WriteObservation(m_ObservationWriter, m_Index);
                
            }
        }
    }
}
