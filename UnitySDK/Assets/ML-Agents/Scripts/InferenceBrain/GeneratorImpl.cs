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
    public class BiDimensionalOutputGenerator : TensorGenerator.Generator
    {
        private ITensorAllocator _allocator;

        public BiDimensionalOutputGenerator(ITensorAllocator allocator)
        {
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the BatchSize input : Will be a one dimensional
    /// integer array of size 1 containing the batch size.
    /// </summary>
    public class BatchSizeGenerator : TensorGenerator.Generator
    {
        private ITensorAllocator _allocator;

        public BatchSizeGenerator(ITensorAllocator allocator)
        {
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensorProxy.Data = _allocator.Alloc(new TensorShape(1,1));
            tensorProxy.Data[0] = batchSize;
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the SequenceLength input : Will be a one
    /// dimensional integer array of size 1 containing 1.
    /// Note : the sequence length is always one since recurrent networks only predict for
    /// one step at the time.
    /// </summary>
    public class SequenceLengthGenerator : TensorGenerator.Generator
    {
        private ITensorAllocator _allocator;

        public SequenceLengthGenerator(ITensorAllocator allocator)
        {
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensorProxy.Shape = new long[0];
            tensorProxy.Data = _allocator.Alloc(new TensorShape(1,1));

            tensorProxy.Data[0] = 1;
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the VectorObservation input : Will be a two
    /// dimensional float array of dimension [batchSize x vectorObservationSize].
    /// It will use the Vector Observation data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    public class VectorObservationGenerator : TensorGenerator.Generator
    {
        private ITensorAllocator _allocator;
        public VectorObservationGenerator(ITensorAllocator allocator)
        {
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
            var vecObsSizeT = tensorProxy.Shape[tensorProxy.Shape.Length - 1];
            
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var vectorObs = agentInfo[agent].stackedVectorObservation;
                for (var j = 0; j < vecObsSizeT; j++)
                {
                    tensorProxy.Data[agentIndex, j] = vectorObs[j];
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
    public class RecurrentInputGenerator : TensorGenerator.Generator
    {
        private ITensorAllocator _allocator;
        
        public RecurrentInputGenerator(ITensorAllocator allocator)
        {
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
            
            var memorySize = tensorProxy.Shape[tensorProxy.Shape.Length - 1];
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
                    tensorProxy.Data[agentIndex, j] = memory[j];
                }
                agentIndex++;
            }
        }
    }
    
    public class BarracudaRecurrentInputGenerator : TensorGenerator.Generator
    {
        private int memoriesCount;
        private int memoryIndex;
        private ITensorAllocator _allocator;   
        
        public BarracudaRecurrentInputGenerator(int memoryIndex, ITensorAllocator allocator)
        {
            this.memoryIndex = memoryIndex;
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
            
            var memorySize = (int)tensorProxy.Shape[tensorProxy.Shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var memory = agentInfo[agent].memories;         

                int offset = memorySize * memoryIndex;
                
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
                    tensorProxy.Data[agentIndex, j] = memory[j + offset];
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
    public class PreviousActionInputGenerator : TensorGenerator.Generator
    {
        private ITensorAllocator _allocator;

        public PreviousActionInputGenerator(ITensorAllocator allocator)
        {
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
            
            var actionSize = tensorProxy.Shape[tensorProxy.Shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var pastAction = agentInfo[agent].storedVectorActions;
                for (var j = 0; j < actionSize; j++)
                {
                    tensorProxy.Data[agentIndex, j] = pastAction[j];
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
    public class ActionMaskInputGenerator : TensorGenerator.Generator
    {
        private ITensorAllocator _allocator;

        public ActionMaskInputGenerator(ITensorAllocator allocator)
        {
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
            
            var maskSize = tensorProxy.Shape[tensorProxy.Shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var maskList = agentInfo[agent].actionMasks;
                for (var j = 0; j < maskSize; j++)
                {
                    var isUnmasked = (maskList != null && maskList[j]) ? 0.0f : 1.0f;
                    tensorProxy.Data[agentIndex, j] = isUnmasked;
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
    public class RandomNormalInputGenerator : TensorGenerator.Generator
    {
        private RandomNormal _randomNormal;
        private ITensorAllocator _allocator;
        
        public RandomNormalInputGenerator(int seed, ITensorAllocator allocator)
        {
            _randomNormal = new RandomNormal(seed);
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
            _randomNormal.FillTensor(tensorProxy);
        }
    }

    /// <summary>
    /// Generates the Tensor corresponding to the Visual Observation input : Will be a 4
    /// dimensional float array of dimension [batchSize x width x heigth x numChannels].
    /// It will use the Texture input data contained in the agentInfo to fill the data
    /// of the tensor.
    /// </summary>
    public class VisualObservationInputGenerator : TensorGenerator.Generator
    {
        private int _index;
        private bool _grayScale;
        private ITensorAllocator _allocator;
        
        public VisualObservationInputGenerator(int index, bool grayScale, ITensorAllocator allocator)
        {
            _index = index;
            _grayScale = grayScale;
            _allocator = allocator;
        }
        
        public void Generate(TensorProxy tensorProxy, int batchSize, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var textures = agentInfo.Keys.Select(
                agent => agentInfo[agent].visualObservations[_index]).ToList();
            
            TensorUtils.ResizeTensor(tensorProxy, batchSize, _allocator);
            Utilities.TextureToTensorProxy(tensorProxy, textures, _grayScale, _allocator);
        } 
    } 
}
