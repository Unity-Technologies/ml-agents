using System.Collections.Generic;
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
        public void Apply(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataAction = tensor.Data as float[,];
            var actionSize = tensor.Shape[tensor.Shape.Length - 1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var action = new float[actionSize];
                for (var j = 0; j < actionSize; j++)
                {
                    action[j] = tensorDataAction[agentIndex, j];
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
        
        public DiscreteActionOutputApplier(int[] actionSize, int seed)
        {
            _actionSize = actionSize;
            _multinomial = new Multinomial(seed);
        }
        
        public void Apply(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataProbabilities = tensor.Data as float[,];
            var batchSize = agentInfo.Keys.Count;
            var actions = new float[batchSize, _actionSize.Length];
            var startActionIndices = Utilities.CumSum(_actionSize);
            for (var actionIndex=0; actionIndex < _actionSize.Length; actionIndex++)
            {
                var nBranchAction = _actionSize[actionIndex];
                var actionProbs = new float[batchSize, nBranchAction];
                for (var batchIndex = 0; batchIndex < batchSize; batchIndex++)
                {
                    for (var branchActionIndex = 0; 
                        branchActionIndex < nBranchAction; 
                        branchActionIndex++)
                    {
                        actionProbs[batchIndex, branchActionIndex] = 
                            tensorDataProbabilities[
                                batchIndex, startActionIndices[actionIndex] + branchActionIndex];
                    }
                }
                var inputTensor = new Tensor()
                {
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, _actionSize[actionIndex]},
                    Data = actionProbs
                };
                var outputTensor = new Tensor()
                {
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, 1},
                    Data = new float[batchSize, 1]
                };
                _multinomial.Eval(inputTensor, outputTensor);
                var outTensor = outputTensor.Data as float[,];
                for (var ii = 0; ii < batchSize; ii++)
                {
                    actions[ii, actionIndex] = outTensor[ii, 0];
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
        private bool firstHalf = true;

        public BarracudaMemoryOutputApplier(bool firstHalf)
        {
            this.firstHalf = firstHalf;
        }
        
        public void Apply(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataMemory = tensor.Data as float[,];
            var agentIndex = 0;
            var memorySize = tensor.Shape[tensor.Shape.Length - 1];
            foreach (var agent in agentInfo.Keys)
            {
                var memory = new List<float>();
                for (var j = 0; j < memorySize; j++)
                {
                    memory.Add(tensorDataMemory[agentIndex, j]);
                }

                if (firstHalf)
                {
                    agent.UpdateMemoriesAction(memory);
                }
                else
                {
                    agent.AppendMemoriesAction(memory);
                }
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
        public void Apply(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataMemory = tensor.Data as float[,];
            var agentIndex = 0;
            var memorySize = tensor.Shape[tensor.Shape.Length - 1];
            foreach (var agent in agentInfo.Keys)
            {
                var memory = new List<float>();
                for (var j = 0; j < memorySize; j++)
                {
                    memory.Add(tensorDataMemory[agentIndex, j]);
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
        public void Apply(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataValue = tensor.Data as float[,];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                agent.UpdateValueAction(tensorDataValue[agentIndex, 0]);
                agentIndex++;
            }
        }
    }
}
