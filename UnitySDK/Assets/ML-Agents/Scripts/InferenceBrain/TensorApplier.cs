using UnityEngine.MachineLearning.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine.Util;

namespace MLAgents.InferenceBrain
{
    public interface TensorApplier
    {
        void Execute(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo);
    }

    /// <summary>
    /// The Applier for the Continuous Action output tensor.
    /// </summary>
    public class ContinuousActionOutputApplier : TensorApplier
    {
        public void Execute(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataAction = tensor.Data as float[,];
            var actionSize = tensor.Shape[1];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var a = new float[actionSize];
                for (var j = 0; j < actionSize; j++)
                {
                    a[j] = tensorDataAction[agentIndex, j];
                }

                agent.UpdateVectorAction(a);
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// The Applier for the Discrete Action output tensor. Uses multinomial to sample discrete
    /// actions from the logits contained in the tensor.
    /// </summary>
    public class DiscreteActionOutputApplier : TensorApplier
    {
        private int[] _actionSize;
        private Multinomial _multinomial;
        
        public DiscreteActionOutputApplier(int[] actionSize, int seed)
        {
            _actionSize = actionSize;
            _multinomial = new Multinomial(seed);
        }
        
        public void Execute(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataProbabilities = tensor.Data as float[,];
            var batchSize = agentInfo.Keys.Count;
            var actions = new float[batchSize, _actionSize.Length];
            var startActionIndices = Utilities.CumSum(_actionSize);
            for (var actionIndex=0; actionIndex < _actionSize.Length; actionIndex++)
            {
                var nBranchAction = startActionIndices[actionIndex + 1] -
                                    startActionIndices[actionIndex];
                var actionProbs = new float[batchSize, nBranchAction];
                for (var ii = 0; ii < batchSize; ii++)
                {
                    for (var jj = 0; jj < nBranchAction; jj++)
                    {
                        actionProbs[ii, jj] = 
                            tensorDataProbabilities[ii, startActionIndices[actionIndex] + jj];
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
                var a = new float[_actionSize.Length];
                for (var j = 0; j < _actionSize.Length; j++)
                {
                    a[j] = actions[agentIndex, j];
                }
                agent.UpdateVectorAction(a);
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// The Applier for the Memory output tensor.
    /// </summary>
    public class MemoryOutputApplier : TensorApplier
    {
        public void Execute(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
        {
            var tensorDataMemory = tensor.Data as float[,];
            var agentIndex = 0;
            var memorySize = tensor.Shape[1];
            foreach (var agent in agentInfo.Keys)
            {
                var a = new List<float>();
                for (var j = 0; j < memorySize; j++)
                {
                    a.Add(tensorDataMemory[agentIndex, j]);
                }

                agent.UpdateMemoriesAction(a);
                agentIndex++;
            }
        }
    }

    /// <summary>
    /// The Applier for the Value Estimate output tensor.
    /// </summary>
    public class ValueEstimateApplier : TensorApplier
    {
        public void Execute(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo)
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
