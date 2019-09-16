using System.Collections.Generic;
using System;
using System.Linq;
using Barracuda;
using MLAgents.InferenceBrain.Utils;

namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// Reshapes a Tensor so that its first dimension becomes equal to the current batch size
    /// and initializes its content to be zeros. Will only work on 2-dimensional tensors.
    /// The second dimension of the Tensor will not be modified.
    /// </summary>
    public class BiDimensionalOutputGenerator : TensorGenerator.IGenerator
    {
        private readonly ITensorAllocator m_Allocator;

        public BiDimensionalOutputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the BatchSize input : Will be a one dimensional
    /// integer array of size 1 containing the batch size.
    /// </summary>
    public class BatchSizeGenerator : TensorGenerator.IGenerator
    {
        private readonly ITensorAllocator m_Allocator;

        public BatchSizeGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensorProxy.data?.Dispose();
            tensorProxy.data = m_Allocator.Alloc(new TensorShape(1, 1));
            tensorProxy.data[0] = batchSize;
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the SequenceLength input : Will be a one
    /// dimensional integer array of size 1 containing 1.
    /// Note : the sequence length is always one since recurrent networks only predict for
    /// one step at the time.
    /// </summary>
    public class SequenceLengthGenerator : TensorGenerator.IGenerator
    {
        private readonly ITensorAllocator m_Allocator;

        public SequenceLengthGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensorProxy.shape = new long[0];
            tensorProxy.data?.Dispose();
            tensorProxy.data = m_Allocator.Alloc(new TensorShape(1, 1));
            tensorProxy.data[0] = 1;
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the VectorObservation input : Will be a two
    /// dimensional float array of dimension [batchSize x vectorObservationSize].
    /// It will use the Vector Observation data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    public class VectorObservationGenerator : TensorGenerator.IGenerator
    {
        private readonly ITensorAllocator m_Allocator;
        public VectorObservationGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            var vecObsSizeT = tensorProxy.shape[tensorProxy.shape.Length - 1];

            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var vectorObs = agentInfo[agent].stackedVectorObservation;
                for (var j = 0; j < vecObsSizeT; j++)
                {
                    tensorProxy.data[agentIndex, j] = vectorObs[j];
                }
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Recurrent input : Will be a two
    /// dimensional float array of dimension [batchSize x memorySize].
    /// It will use the Memory data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    public class RecurrentInputGenerator : TensorGenerator.IGenerator
    {
        private readonly ITensorAllocator m_Allocator;

        public RecurrentInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var memorySize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var memory = agentInfo[agent].memories;
                if (memory == null)
                {
                    agentIndex++;
                    continue;
                }
                for (var j = 0; j < Math.Min(memorySize, memory.Count); j++)
                {
                    if (j >= memory.Count)
                    {
                        break;
                    }
                    tensorProxy.data[agentIndex, j] = memory[j];
                }
                agentIndex++;
            }
        }
    }

    public class BarracudaRecurrentInputGenerator : TensorGenerator.IGenerator
    {
        private int m_MemoriesCount;
        private readonly int m_MemoryIndex;
        private readonly ITensorAllocator m_Allocator;

        public BarracudaRecurrentInputGenerator(int memoryIndex, ITensorAllocator allocator)
        {
            m_MemoryIndex = memoryIndex;
            m_Allocator = allocator;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var memorySize = (int)tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var memory = agentInfo[agent].memories;

                var offset = memorySize * m_MemoryIndex;

                if (memory == null)
                {
                    agentIndex++;
                    continue;
                }
                for (var j = 0; j < memorySize; j++)
                {
                    if (j >= memory.Count)
                    {
                        break;
                    }
                    tensorProxy.data[agentIndex, j] = memory[j + offset];
                }
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Previous Action input : Will be a two
    /// dimensional integer array of dimension [batchSize x actionSize].
    /// It will use the previous action data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    public class PreviousActionInputGenerator : TensorGenerator.IGenerator
    {
        private readonly ITensorAllocator m_Allocator;

        public PreviousActionInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var actionSize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var pastAction = agentInfo[agent].storedVectorActions;
                for (var j = 0; j < actionSize; j++)
                {
                    tensorProxy.data[agentIndex, j] = pastAction[j];
                }

                agentIndex++;
            }
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Action Mask input : Will be a two
    /// dimensional float array of dimension [batchSize x numActionLogits].
    /// It will use the Action Mask data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    public class ActionMaskInputGenerator : TensorGenerator.IGenerator
    {
        private readonly ITensorAllocator m_Allocator;

        public ActionMaskInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var maskSize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var maskList = agentInfo[agent].actionMasks;
                for (var j = 0; j < maskSize; j++)
                {
                    var isUnmasked = (maskList != null && maskList[j]) ? 0.0f : 1.0f;
                    tensorProxy.data[agentIndex, j] = isUnmasked;
                }
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Epsilon input : Will be a two
    /// dimensional float array of dimension [batchSize x actionSize].
    /// It will use the generate random input data from a normal Distribution.
    /// </summary>
    public class RandomNormalInputGenerator : TensorGenerator.IGenerator
    {
        private readonly RandomNormal m_RandomNormal;
        private readonly ITensorAllocator m_Allocator;

        public RandomNormalInputGenerator(int seed, ITensorAllocator allocator)
        {
            m_RandomNormal = new RandomNormal(seed);
            m_Allocator = allocator;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            TensorUtils.FillTensorWithRandomNormal(tensorProxy, m_RandomNormal);
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Visual Observation input : Will be a 4
    /// dimensional float array of dimension [batchSize x width x height x numChannels].
    /// It will use the Texture input data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    public class VisualObservationInputGenerator : TensorGenerator.IGenerator
    {
        private readonly int m_Index;
        private readonly bool m_GrayScale;
        private readonly ITensorAllocator m_Allocator;

        public VisualObservationInputGenerator(
            int index, bool grayScale, ITensorAllocator allocator)
        {
            m_Index = index;
            m_GrayScale = grayScale;
            m_Allocator = allocator;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var textures = agentInfo.Keys.Select(
                agent => agentInfo[agent].visualObservations[m_Index]).ToList();

            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            Utilities.TextureToTensorProxy(textures, tensorProxy, m_GrayScale);
        }
    }
}
