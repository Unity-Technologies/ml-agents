using UnityEngine;
using UnityEngine.MachineLearning.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using System.Linq;
using System;

namespace MLAgents
{
    public class InternalBrainTensorApplier : 
        Dictionary<string, Action<Tensor, Dictionary<Agent, AgentInfo>>>
    {
        Dictionary<string, Action<Tensor, Dictionary<Agent, AgentInfo>>>  dict;
        
         public InternalBrainTensorApplier(
            CoreBrainInternal.NodeNames _nodeNames, 
            BrainParameters bp,
            Multinomial multinomial)
        {
            dict = new Dictionary<string, Action<Tensor, Dictionary<Agent, AgentInfo>>>();
            
            dict[_nodeNames.ValueEstimateOutput] = ApplyValueEstimate;
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                dict[_nodeNames.ActionOutput] = ApplyContinuousActionOutput;
            }
            else
            {
                dict[_nodeNames.ActionOutput] = (tensor, agentInfo) =>
                    ApplyDiscreteActionOutput(tensor, agentInfo, multinomial,
                        bp.vectorActionSize);
            }
            dict[_nodeNames.RecurrentOutOutput] = ApplyMemoryOutput;


        }
        
        public new Action<Tensor, Dictionary<Agent, AgentInfo>> this[string index]
        {
            get
            {
                return dict[index];
            }

            set
            {
                dict[index] = value;
            }
        }

        public new bool ContainsKey(string key)
        {
            return dict.ContainsKey(key);
        }
        
        
        private static void ApplyContinuousActionOutput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo)
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

        /// <summary>
        /// Generates an array containing the starting indicies of each branch in the vector action
        /// Makes a cumulative sum.
        /// </summary>
        /// <returns></returns>
        private static int[] CreateActionStartinIndices(int[] vectorActionSize)
        {
            var runningSum = 0;
            var result = new int[vectorActionSize.Length + 1];
            for (var actionIndex = 0;
                actionIndex < vectorActionSize.Length; actionIndex++)
            {
                runningSum += vectorActionSize[actionIndex];
                result[actionIndex + 1] = runningSum;
            }
            return result;
        }
        
        private static void ApplyDiscreteActionOutput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo,
            Multinomial multinomial,
            int[] actionSize)
        {
            var tensorDataProbabilities = tensor.Data as float[,];
            var batchSize = agentInfo.Keys.Count;
            var actions = new float[batchSize, actionSize.Length];
            var startActionIndices = CreateActionStartinIndices(actionSize);
            
            for (var actionIndex=0; actionIndex < actionSize.Length; actionIndex++)
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
                    ValueType= Tensor.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, actionSize[actionIndex]},
                    Data = actionProbs
                };
                var outputTensor = new Tensor()
                {
                    ValueType= Tensor.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, 1},
                    Data = new float[batchSize, 1]
                };
                multinomial.Eval(inputTensor, outputTensor);
                var outTensor = outputTensor.Data as float[,];
                for (var ii = 0; ii < batchSize; ii++)
                {
                    actions[ii, actionIndex] = outTensor[ii, 0];
                }
            }
            
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var a = new float[actionSize.Length];
                for (var j = 0; j < actionSize.Length; j++)
                {
                    a[j] = actions[agentIndex, j];
                }
    
                agent.UpdateVectorAction(a);
                agentIndex++;
            }
        }

        private static void ApplyMemoryOutput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo)
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

                // TODO Make better
                agent.UpdateMemoriesAction(a);
                agentIndex++;
            }
        }

        private static void ApplyValueEstimate(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo)
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