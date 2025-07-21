using System.Collections.Generic;
using System;
using Unity.InferenceEngine;
using Unity.MLAgents.Inference.Utils;
using Unity.MLAgents.Sensors;
using static Unity.MLAgents.Inference.TensorProxy;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Reshapes a Tensor so that its first dimension becomes equal to the current batch size
    /// and initializes its content to be zeros. Will only work on 2-dimensional tensors.
    /// The second dimension of the Tensor will not be modified.
    /// </summary>
    internal class BiDimensionalOutputGenerator : TensorGenerator.IGenerator
    {
        public BiDimensionalOutputGenerator() { }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize);
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the BatchSize input : Will be a one dimensional
    /// integer array of size 1 containing the batch size.
    /// </summary>
    internal class BatchSizeGenerator : TensorGenerator.IGenerator
    {
        public BatchSizeGenerator() { }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            tensorProxy.data?.Dispose();
            var newTensorShape = new TensorShape(1, 1);
            tensorProxy.data = TensorUtils.CreateEmptyTensor(newTensorShape, tensorProxy.DType);
            tensorProxy.data.CompleteAllPendingOperations();

            ((Tensor<int>)tensorProxy.data)[0] = batchSize;
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
        public SequenceLengthGenerator() { }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            tensorProxy.shape = Array.Empty<int>();
            tensorProxy.data?.Dispose();
            var newTensorShape = new TensorShape(1, 1);
            tensorProxy.data = TensorUtils.CreateEmptyTensor(newTensorShape, tensorProxy.DType);
            tensorProxy.data.CompleteAllPendingOperations();

            ((Tensor<int>)tensorProxy.data)[0] = 1;
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
        Dictionary<int, List<float>> m_Memories;

        public RecurrentInputGenerator(
            Dictionary<int, List<float>> memories)
        {
            m_Memories = memories;
        }

        public void Generate(
            TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize);

            var memorySize = tensorProxy.data.Width();

            tensorProxy.data.CompleteAllPendingOperations();

            var agentIndex = 0;

            for (var infoIndex = 0; infoIndex < infos.Count; infoIndex++)
            {
                var infoSensorPair = infos[infoIndex];
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
                        ((Tensor<float>)tensorProxy.data)[agentIndex, 0, j] = 0;
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

                    ((Tensor<float>)tensorProxy.data)[agentIndex, 0, j] = memory[j];
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
        public PreviousActionInputGenerator() { }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize);
            tensorProxy.data.CompleteAllPendingOperations();

            var actionSize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            for (var infoIndex = 0; infoIndex < infos.Count; infoIndex++)
            {
                var infoSensorPair = infos[infoIndex];
                var info = infoSensorPair.agentInfo;
                var pastAction = info.storedActions.DiscreteActions;
                if (!pastAction.IsEmpty())
                {
                    for (var j = 0; j < actionSize; j++)
                    {
                        ((Tensor<int>)tensorProxy.data)[agentIndex, j] = pastAction[j];
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
        public ActionMaskInputGenerator() { }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize);

            tensorProxy.data.CompleteAllPendingOperations();

            var maskSize = tensorProxy.shape[tensorProxy.shape.Length - 1];
            var agentIndex = 0;
            for (var infoIndex = 0; infoIndex < infos.Count; infoIndex++)
            {
                var infoSensorPair = infos[infoIndex];
                var agentInfo = infoSensorPair.agentInfo;
                var maskList = agentInfo.discreteActionMasks;

                for (var j = 0; j < maskSize; j++)
                {
                    var isUnmasked = (maskList != null && maskList[j]) ? 0.0f : 1.0f;
                    ((Tensor<float>)tensorProxy.data)[agentIndex, j] = isUnmasked;
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

        public RandomNormalInputGenerator(int seed)
        {
            m_RandomNormal = new RandomNormal(seed);
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize);
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
        List<int> m_SensorIndices = new List<int>();
        ObservationWriter m_ObservationWriter = new ObservationWriter();

        public ObservationGenerator() { }

        public void AddSensorIndex(int sensorIndex)
        {
            m_SensorIndices.Add(sensorIndex);
        }

        public void Generate(TensorProxy tensorProxy, int batchSize, IList<AgentInfoSensorsPair> infos)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize);
            var agentIndex = 0;
            for (var infoIndex = 0; infoIndex < infos.Count; infoIndex++)
            {
                var info = infos[infoIndex];
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
                    for (var sensorIndexIndex = 0; sensorIndexIndex < m_SensorIndices.Count; sensorIndexIndex++)
                    {
                        var sensorIndex = m_SensorIndices[sensorIndexIndex];
                        var sensor = info.sensors[sensorIndex];
                        m_ObservationWriter.SetTarget(tensorProxy, agentIndex, tensorOffset);
                        var numWritten = sensor.Write(m_ObservationWriter);
                        tensorOffset += numWritten;
                    }
                }

                agentIndex++;
            }
        }
    }
}
