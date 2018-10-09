using UnityEngine.MachineLearning.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using System.Linq;
using System;

namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// Mapping between Tensor names and generators.
    /// A TensorGenerator implements a Dictionary of strings (node names) to an Action.
    /// The Action take as argument the tensor, the current batch size and a Dictionary of
    /// Agent to AgentInfo corresponding to the current batch.
    /// Each Generator reshapes and fills the data of the tensor based of the data of the batch.
    /// When the Tensor is an Input to the model, the shape of the Tensor will be modified
    /// depending on the current batch size and the data of the Tensor will be filled using the
    /// Dictionary of Agent to AgentInfo.
    /// When the Tensor is an Output of the model, only the shape of the Tensor will be modified
    /// using the current batch size. The data will be prefilled with zeros.
    /// </summary>
    public class TensorGenerators
    {
        Dictionary<string, Action<Tensor, int, Dictionary<Agent, AgentInfo>>>  dict = 
            new Dictionary<string, Action<Tensor, int, Dictionary<Agent, AgentInfo>>>();
        
        /// <summary>
        /// The constructor for the TensorGenerators. Returns a new TensorGenerators object.
        /// </summary>
        /// <param name="bp"> The BrainParameters used to determines what Generators will be
        /// used</param>
        /// <param name="randomNormal"> The RandomNormal object some of the Generators will
        /// be initialized with.</param>
        public TensorGenerators(BrainParameters bp, RandomNormal randomNormal)
        {
            // Generator for Inputs
            dict[TensorNames.BatchSizePlaceholder] = GenerateBatchSize;
            dict[TensorNames.SequenceLengthPlaceholder] = GenerateSequenceLength;
            dict[TensorNames.VectorObservationPlacholder] = GenerateVectorObservation;
            dict[TensorNames.RecurrentInPlaceholder] = GenerateRecurrentInput;
            dict[TensorNames.PreviousActionPlaceholder] = GeneratePreviousActionInput;
            dict[TensorNames.ActionMaskPlaceholder] = GenerateActionMaskInput;
            dict[TensorNames.RandomNormalEpsilonPlaceholder] =
                (tensor, batchSize, agentInfo) =>
                    GenerateRandomNormalInput(tensor, batchSize, agentInfo, randomNormal);
            if (bp.cameraResolutions != null)
            {
                for (var visIndex = 0;
                    visIndex < bp.cameraResolutions.Length;
                    visIndex++)
                {
                    var index = visIndex;
                    var bw = bp.cameraResolutions[visIndex].blackAndWhite;
                    dict[TensorNames.VisualObservationPlaceholderPrefix + visIndex] =
                        (tensor, batchSize, agentInfo) =>
                            GenerateVisualObservationInput(tensor, agentInfo, index, bw);
                }
            }
            // Generators for Outputs
            dict[TensorNames.ActionOutput] = ReshapeBiDimensionalOutput;
            dict[TensorNames.RecurrentOutput] = ReshapeBiDimensionalOutput;
            dict[TensorNames.ValueEstimateOutput] = ReshapeBiDimensionalOutput;
        }
        
        /// <summary>
        /// Access the Generator corresponding to the key index
        /// </summary>
        /// <param name="index">The tensor name of the tensor</param>
        public Action<Tensor, int, Dictionary<Agent, AgentInfo>> this[string index]
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
        /// Evaluates whether the tensor name has a Generator
        /// </summary>
        /// <param name="key">The tensor name of the tensor</param>
        /// <returns>true if key is in the TensorGenerators, false otherwise</returns>
        public bool ContainsKey(string key)
        {
            return dict.ContainsKey(key);
        }
        
        /// <summary>
        /// Reshapes a Tensor so that his first diemsion becomes equal to the current batch size
        /// and initializes its content to be zeros. Will only work on 2-dimensional tensors.
        /// The second dimension of the Tensor will not be modiied.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="batchSize"></param>
        /// <param name="agentInfo"></param>
        private static void ReshapeBiDimensionalOutput(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            var shapeSecondAxis = tensor.Shape[1];
            tensor.Shape[0] = batchSize;
            if (tensor.ValueType == Tensor.TensorType.FloatingPoint)
            {
                tensor.Data = new float[batchSize, shapeSecondAxis];
            }
            else
            {
                tensor.Data = new int[batchSize, shapeSecondAxis];
            }
        }
        
        /// <summary>
        /// Generates the Tensor corresponding to the BatchSize input : Will be a one dimensional
        /// integer array of size 1 containing the batch size.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="batchSize"> The number of agents present in the current batch</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void GenerateBatchSize(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Data = new int[] {batchSize};
        }
        
        /// <summary>
        /// Generates the Tensor corresponding to the SequenceLength input : Will be a one
        /// dimensional integer array of size 1 containing 1.
        /// Note : the sequence length is always one since recurrent networks only predicts for
        /// one step at the time.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="batchSize"> The number of agents present in the current batch</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void GenerateSequenceLength(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Data = new int[] {1};
        }
        
        /// <summary>
        /// Generates the Tensor corresponding to the VectorObservation input : Will be a two
        /// dimensional float array of dimension [batchSize x vectorObservationSize].
        /// It will use the Vector Observation data contained in the agentInfo to fill the data
        /// of the tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="batchSize"> The number of agents present in the current batch</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void GenerateVectorObservation(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Shape[0] = batchSize;
            var vecObsSizeT = tensor.Shape[1];
            tensor.Data = new float[batchSize, vecObsSizeT];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var vectorObs = agentInfo[agent].stackedVectorObservation;
                for (var j = 0; j < vecObsSizeT; j++)
                {
                    tensor.Data.SetValue(vectorObs[j], new int[2] {agentIndex, j});
                }
                agentIndex++;
            }
        }
        
        /// <summary>
        /// Generates the Tensor corresponding to the Recurrent input : Will be a two
        /// dimensional float array of dimension [batchSize x memorySize].
        /// It will use the Memory data contained in the agentInfo to fill the data
        /// of the tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="batchSize"> The number of agents present in the current batch</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void GenerateRecurrentInput(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Shape[0] = batchSize;
            var memorySize = tensor.Shape[1];
            tensor.Data = new float[batchSize, memorySize];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var memory = agentInfo[agent].memories;
                for (var j = 0; j < memorySize; j++)
                {
                    if (memory == null)
                    {
                        break;
                    }
                    if (j >= memory.Count)
                    {
                        break;
                    }
                    tensor.Data.SetValue(memory[j], new int[2] {agentIndex, j});
                }
                agentIndex++;
            }
        }
        
        /// <summary>
        /// Generates the Tensor corresponding to the Previous Action input : Will be a two
        /// dimensional integer array of dimension [batchSize x actionSize].
        /// It will use the previous action data contained in the agentInfo to fill the data
        /// of the tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="batchSize"> The number of agents present in the current batch</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void GeneratePreviousActionInput(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            if (tensor.ValueType != Tensor.TensorType.Integer)
            {
                throw new NotImplementedException(
                    "Previous Action Inputs are only valid for discrete control");
            }
            tensor.Shape[0] = batchSize;
            var actionSize = tensor.Shape[1];
            tensor.Data = new int[batchSize, actionSize];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var pastAction = agentInfo[agent].storedVectorActions;
                for (var j = 0; j < actionSize; j++)
                {
                    tensor.Data.SetValue((int)pastAction[j], new int[2] {agentIndex, j});
                }
                agentIndex++;
            }
        }
        
        /// <summary>
        /// Generates the Tensor corresponding to the Action Mask input : Will be a two
        /// dimensional float array of dimension [batchSize x numActionLogits].
        /// It will use the Action Mask data contained in the agentInfo to fill the data
        /// of the tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="batchSize"> The number of agents present in the current batch</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        private static void GenerateActionMaskInput(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Shape[0] = batchSize;
            var maskSize = tensor.Shape[1];
            tensor.Data = new float[batchSize, maskSize];
            var agentIndex = 0;
            foreach (var agent in agentInfo.Keys)
            {
                var maskList = agentInfo[agent].actionMasks;
                for (var j = 0; j < maskSize; j++)
                {
                    if (maskList != null)
                    {
                        tensor.Data.SetValue(
                            maskList[j] ? 0.0f : 1.0f,
                            new int[2] {agentIndex, j});
                    }
                    else
                    {
                        tensor.Data.SetValue(
                            1.0f,
                            new int[2] {agentIndex, j});
                    }
                }
                agentIndex++;
            }
        }

        /// <summary>
        /// Generates the Tensor corresponding to the Epsilon input : Will be a two
        /// dimensional float array of dimension [batchSize x actionSize].
        /// It will use the generate random input data using a RandomNormal Distribution.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="batchSize"> The number of agents present in the current batch</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        /// <param name="randomNormal"> The RandomNormal object that will be used to generate the
        /// input.
        /// </param>
        private static void GenerateRandomNormalInput(Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo,
            RandomNormal randomNormal)
        {
            tensor.Shape[0] = batchSize;
            var actionSize = tensor.Shape[1];
            tensor.Data = new float[batchSize, actionSize];
            randomNormal.FillTensor(tensor);
        }

        /// <summary>
        /// Generates the Tensor corresponding to the Visual Observation input : Will be a 4
        /// dimensional float array of dimension [batchSize x width x heigth x numChannels].
        /// It will use the Texture input data contained in the agentInfo to fill the data
        /// of the tensor.
        /// </summary>
        /// <param name="tensor"> The tensor containing the data to be appied to the Agents</param>
        /// <param name="agentInfo"> The Dictionary of Agent to AgentInfo of the current batch
        /// </param>
        /// <param name="visIndex"> The index of the Visual Observation in the Agent's data</param>
        /// <param name="bw"> If true, the Tensor will have only one channel (The observation
        /// is grayscake), if false, the Tensor will have 3 channels (the Observation hads three
        /// channels corresponding to RGB).</param>
        private static void GenerateVisualObservationInput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo,
            int visIndex,
            bool bw)
        {
            var textures = agentInfo.Keys.Select(
                agent => agentInfo[agent].visualObservations[visIndex]).ToList();
            tensor.Data = Utilities.TextureToFloatArray(textures, bw);
        }      
    }
}
