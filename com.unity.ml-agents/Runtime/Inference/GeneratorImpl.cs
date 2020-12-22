using System.Collections.Generic;
using System;
using UnityEngine;
using Unity.Barracuda;
using Unity.MLAgents.Inference.Utils;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Inference
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

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
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

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
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
    internal class SequenceLengthGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public SequenceLengthGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
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
    internal class VectorObservationGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;
        List<int> m_SensorIndices = new List<int>();
        ObservationWriter m_ObservationWriter = new ObservationWriter();

        public VectorObservationGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void AddSensorIndex(int sensorIndex)
        {
            m_SensorIndices.Add(sensorIndex);
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            var vecObsSizeT = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var info in infos)
            {
                if (info.agentInfo.done)
                {
                    // If the agent is done, we might have a stale reference to the sensors
                    // e.g. a dependent object might have been disposed.
                    // To avoid this, just fill observation with zeroes instead of calling sensor.Write.
                    TensorUtils.FillTensorBatch(tensorProxy, agentIndex, 0.0f);
                }
                else
                {
                    var tensorOffset = 0;
                    // Write each sensor consecutively to the tensor
                    foreach (var sensorIndex in m_SensorIndices)
                    {
                        var sensor = info.sensors[sensorIndex];
                        m_ObservationWriter.SetTarget(tensorProxy, agentIndex, tensorOffset);
                        var numWritten = sensor.Write(m_ObservationWriter);
                        tensorOffset += numWritten;
                    }
                    Debug.AssertFormat(
                        tensorOffset == vecObsSizeT,
                        "mismatch between vector observation size ({0}) and number of observations written ({1})",
                        vecObsSizeT, tensorOffset
                    );
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
    internal class RecurrentInputGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;
        Dictionary<int, List<float>> m_Memories;

        public RecurrentInputGenerator(
            ITensorAllocator allocator,
            Dictionary<int, List<float>> memories)
        {
            m_Allocator = allocator;
            m_Memories = memories;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var memorySize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var infoSensorPair in infos)
            {
                var info = infoSensorPair.agentInfo;
                List<float> memory;

                if (info.done)
                {
                    m_Memories.Remove(info.episodeId);
                }
                if (!m_Memories.TryGetValue(info.episodeId, out memory))
                {
                    for (var j = 0; j < memorySize; j++)
                    {
                        tensorProxy.data[agentIndex, j] = 0;
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
                    tensorProxy.data[agentIndex, j] = memory[j];
                }
                agentIndex++;
            }
        }
    }

    internal class BarracudaRecurrentInputGenerator : TensorGenerator.IGenerator
    {
        int m_MemoriesCount;
        readonly int m_MemoryIndex;
        readonly ITensorAllocator m_Allocator;

        Dictionary<int, List<float>> m_Memories;

        public BarracudaRecurrentInputGenerator(
            int memoryIndex,
            ITensorAllocator allocator,
            Dictionary<int, List<float>> memories)
        {
            m_MemoryIndex = memoryIndex;
            m_Allocator = allocator;
            m_Memories = memories;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var memorySize = (int)tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var infoSensorPair in infos)
            {
                var info = infoSensorPair.agentInfo;
                var offset = memorySize * m_MemoryIndex;
                List<float> memory;
                if (info.done)
                {
                    m_Memories.Remove(info.episodeId);
                }
                if (!m_Memories.TryGetValue(info.episodeId, out memory))
                {
                    for (var j = 0; j < memorySize; j++)
                    {
                        tensorProxy.data[agentIndex, j] = 0;
                    }
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
    internal class PreviousActionInputGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public PreviousActionInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var actionSize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var infoSensorPair in infos)
            {
                var info = infoSensorPair.agentInfo;
                var pastAction = info.storedActions.DiscreteActions;
                if (!pastAction.IsEmpty())
                {
                    for (var j = 0; j < actionSize; j++)
                    {
                        tensorProxy.data[agentIndex, j] = pastAction[j];
                    }
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
    internal class ActionMaskInputGenerator : TensorGenerator.IGenerator
    {
        readonly ITensorAllocator m_Allocator;

        public ActionMaskInputGenerator(ITensorAllocator allocator)
        {
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);

            var maskSize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            foreach (var infoSensorPair in infos)
            {
                var agentInfo = infoSensorPair.agentInfo;
                var maskList = agentInfo.discreteActionMasks;
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
    internal class RandomNormalInputGenerator : TensorGenerator.IGenerator
    {
        readonly RandomNormal m_RandomNormal;
        readonly ITensorAllocator m_Allocator;

        public RandomNormalInputGenerator(int seed, ITensorAllocator allocator)
        {
            m_RandomNormal = new RandomNormal(seed);
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
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
    internal class VisualObservationInputGenerator : TensorGenerator.IGenerator
    {
        readonly int m_SensorIndex;
        readonly ITensorAllocator m_Allocator;
        ObservationWriter m_ObservationWriter = new ObservationWriter();

        public VisualObservationInputGenerator(
            int sensorIndex, ITensorAllocator allocator)
        {
            m_SensorIndex = sensorIndex;
            m_Allocator = allocator;
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IEnumerable<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, m_Allocator);
            var agentIndex = 0;
            foreach (var infoSensorPair in infos)
            {
                var sensor = infoSensorPair.sensors[m_SensorIndex];
                if (infoSensorPair.agentInfo.done)
                {
                    // If the agent is done, we might have a stale reference to the sensors
                    // e.g. a dependent object might have been disposed.
                    // To avoid this, just fill observation with zeroes instead of calling sensor.Write.
                    TensorUtils.FillTensorBatch(tensorProxy, agentIndex, 0.0f);
                }
                else
                {
                    m_ObservationWriter.SetTarget(tensorProxy, agentIndex, 0);
                    sensor.Write(m_ObservationWriter);
                }
                agentIndex++;
            }
        }
    }
}
