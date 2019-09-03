using System.Collections.Generic;
using System.Linq;
using Barracuda;
using MLAgents.InferenceBrain.Utils;
using UnityEngine;

namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// The Applier for the Continuous Action output tensor. Tensor is assumed to contain the
    /// continuous action data of the agents in the batch.
    /// </summary>
    public class ContinuousActionOutputApplier : TensorApplier.Applier
    {
        public void Apply(TensorProxy tensorProxy, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var actionSize = tensorProxy.Shape[tensorProxy.Shape.Length - 1];    
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var action = new float[actionSize];
                for (var j = 0; j < actionSize; j++)
                {
                    action[j] = tensorProxy.Data[agentIndex, j];
                }
                agent.UpdateVectorAction(action);
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// The Applier for the Discrete Action output tensor. Uses multinomial to sample discrete
    /// actions from the logits contained in the tensor.
    /// </summary>
    public class DiscreteActionOutputApplier : TensorApplier.Applier
    {
        private int[] _actionSize;
        private Multinomial _multinomial;
        private ITensorAllocator _allocator;
        
        public DiscreteActionOutputApplier(int[] actionSize, int seed, ITensorAllocator allocator)
        {
            _actionSize = actionSize;
            _multinomial = new Multinomial(seed);
            _allocator = allocator;
        }
        
        public void Apply(TensorProxy tensorProxy, Dictionary<Agent, AgentInfo> agentInfo)
        {
            //var tensorDataProbabilities = tensorProxy.Data as float[,];
            var batchSize = agentInfo.Keys.Count;
            var actions = new float[batchSize, _actionSize.Length];
            var startActionIndices = Utilities.CumSum(_actionSize);
            for (var actionIndex=0; actionIndex < _actionSize.Length; actionIndex++)
            {
                var nBranchAction = _actionSize[actionIndex];
                var actionProbs = new TensorProxy()
                {
                    ValueType = TensorProxy.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, nBranchAction},
                    Data = _allocator.Alloc(new TensorShape(batchSize, nBranchAction))
                };
                
                for (var batchIndex = 0; batchIndex < batchSize; batchIndex++)
                {
                    for (var branchActionIndex = 0; 
                        branchActionIndex < nBranchAction; 
                        branchActionIndex++)
                    {
                        actionProbs.Data[batchIndex, branchActionIndex] = 
                            tensorProxy.Data[batchIndex, startActionIndices[actionIndex] + branchActionIndex];
                    }
                }
                
                var outputTensor = new TensorProxy()
                {
                    ValueType = TensorProxy.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, 1},
                    Data = _allocator.Alloc(new TensorShape(batchSize, 1))
                };
                
                _multinomial.Eval(actionProbs, outputTensor);
                
                for (var ii = 0; ii < batchSize; ii++)
                {
                    actions[ii, actionIndex] = outputTensor.Data[ii, 0];
                }
            }
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var action = new float[_actionSize.Length];
                for (var j = 0; j < _actionSize.Length; j++)
                {
                    action[j] = actions[agentIndex, j];
                }
                agent.UpdateVectorAction(action);
                agentIndex++;
            }
        }
    }
    
    public class BarracudaMemoryOutputApplier : TensorApplier.Applier
    {
        private int memoriesCount;
        private int memoryIndex;

        public BarracudaMemoryOutputApplier(int memoriesCount, int memoryIndex)
        {
            this.memoriesCount = memoriesCount;
            this.memoryIndex = memoryIndex;
        }
        
        public void Apply(TensorProxy tensorProxy, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var agentIndex = 0;
            var memorySize = (int)tensorProxy.Shape[tensorProxy.Shape.Length - 1];
            
            foreach (var agent in agentInfo.Keys)
            {
                var memory = agent.GetMemoriesAction();

                if (memory == null || memory.Count < memorySize * memoriesCount)
                {
                    memory = new List<float>();
                    memory.AddRange(Enumerable.Repeat(0f, memorySize * memoriesCount));
                }

                for (var j = 0; j < memorySize; j++)
                {
                    memory[memorySize * memoryIndex + j] = tensorProxy.Data[agentIndex, j];
                }
                
                agent.UpdateMemoriesAction(memory);

                agentIndex++;
            }
        }
    }

    /// <summary>
    /// The Applier for the Memory output tensor. Tensor is assumed to contain the new
    /// memory data of the agents in the batch.
    /// </summary>
    public class MemoryOutputApplier : TensorApplier.Applier
    {
        public void Apply(TensorProxy tensorProxy, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var agentIndex = 0;
            var memorySize = tensorProxy.Shape[tensorProxy.Shape.Length - 1];
            foreach (var agent in agentInfo.Keys)
            {
                var memory = new List<float>();
                for (var j = 0; j < memorySize; j++)
                {
                    memory.Add(tensorProxy.Data[agentIndex, j]);
                }

                agent.UpdateMemoriesAction(memory);
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// The Applier for the Value Estimate output tensor. Tensor is assumed to contain the
    /// value estimates of the agents in the batch.
    /// </summary>
    public class ValueEstimateApplier : TensorApplier.Applier
    {
        public void Apply(TensorProxy tensorProxy, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                agent.UpdateValueAction(tensorProxy.Data[agentIndex, 0]);
                agentIndex++;
            }
        }
    }
}
