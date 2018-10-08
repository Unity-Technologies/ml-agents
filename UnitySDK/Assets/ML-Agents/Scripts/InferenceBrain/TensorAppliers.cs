using UnityEngine.MachineLearning.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using System;

namespace MLAgents.InferenceBrain
{    
    /// <summary>
    /// Mapping between the output Tensor names and the method that will use the
    /// output tensors and the Agents present in the batch to update their action, memories and
    /// value estimates.
    /// A TensorApplier implements a Dictionary of strings (node names) to an Action.
    /// This action takes as input the Tensor and the Dictionary of Agent to AgentInfo for
    /// the current batch.
    /// </summary>
    public class TensorAppliers : 
        Dictionary<string, Action<Tensor, Dictionary<Agent, AgentInfo>>>
    {
        Dictionary<string, Action<Tensor, Dictionary<Agent, AgentInfo>>>  dict;

        /// <summary>
        /// Constructor of TensorAppliers. Returns a new TensorAppliers object.
        /// </summary>
        /// <param name="bp"> The BrainParameters used to determines what Appliers will be
        /// used</param>
        /// <param name="multinomial"> The Multinomial objects some of the Appliers will
        /// be initialized with.</param>
        public TensorAppliers(BrainParameters bp, Multinomial multinomial)
        {
            dict = new Dictionary<string, Action<Tensor, Dictionary<Agent, AgentInfo>>>();
            
            dict[TensorNames.ValueEstimateOutput] = ApplyValueEstimate;
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                dict[TensorNames.ActionOutput] = ApplyContinuousActionOutput;
            }
            else
            {
                dict[TensorNames.ActionOutput] = (tensor, agentInfo) =>
                    ApplyDiscreteActionOutput(tensor, agentInfo, multinomial,
                        bp.vectorActionSize);
            }
            dict[TensorNames.RecurrentOutput] = ApplyMemoryOutput;
        }

        /// <summary>
        /// Access the Applier corresponding to the key index
        /// </summary>
        /// <param name="index">The tensor name of the tensor</param>
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

        /// <summary>
        /// Evaluates whether the tensor name has an Applier
        /// </summary>
        /// <param name="key">The tensor name of the tensor</param>
        /// <returns>true if key is in the TensorAppliers, false otherwise</returns>
        public new bool ContainsKey(string key)
        {
            return dict.ContainsKey(key);
        }

        /// <summary>
        /// The Applier for the Continuous Action output tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void ApplyContinuousActionOutput(Tensor tensor,
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
        /// The Applier for the Discrete Action output tensor. Uses multinomial to sample discrete
        /// actions from the logits contained in the tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        /// <param name="multinomial"> The Multinomial object that will be used to sample the
        /// actions</param>
        /// <param name="actionSize"> An array of integers corresponding to the number of actions
        /// possible per branch.</param>
        private static void ApplyDiscreteActionOutput(Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo,
            Multinomial multinomial,
            int[] actionSize)
        {
            var tensorDataProbabilities = tensor.Data as float[,];
            var batchSize = agentInfo.Keys.Count;
            var actions = new float[batchSize, actionSize.Length];
            var startActionIndices = Utilities.CumSum(actionSize);
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
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Shape = new long[]{batchSize, actionSize[actionIndex]},
                    Data = actionProbs
                };
                var outputTensor = new Tensor()
                {
                    ValueType = Tensor.TensorType.FloatingPoint,
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

        /// <summary>
        /// The Applier for the Memory output tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void ApplyMemoryOutput(Tensor tensor,
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
                agent.UpdateMemoriesAction(a);
                agentIndex++;
            }
        }

        /// <summary>
        /// The Applier for the Value Estimate output tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
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
