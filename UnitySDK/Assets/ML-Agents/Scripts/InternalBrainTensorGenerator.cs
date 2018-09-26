using UnityEngine;
using UnityEngine.MachineLearning.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine.Util;
using System.Linq;
using System;
using System.CodeDom;

namespace MLAgents
{
    public class InternalBrainTensorGenerator : 
        Dictionary<string, Action<Tensor, int, Dictionary<Agent, AgentInfo>>> 
    {
        Dictionary<string, Action<Tensor, int, Dictionary<Agent, AgentInfo>>>  dict;
        
        public InternalBrainTensorGenerator(
            CoreBrainInternal.NodeNames nodeNames, 
            BrainParameters bp,
            RandomNormal randomNormal)
        {
            dict = new Dictionary<string, Action<Tensor, int, Dictionary<Agent, AgentInfo>>>();
            dict[nodeNames.BatchSizePlaceholder] = GenerateBatchSize;
            dict[nodeNames.SequenceLengthPlaceholder] = GenerateSequenceLength;
            dict[nodeNames.VectorObservationPlacholder] = GenerateVectorObservation;
            dict[nodeNames.RecurrentInPlaceholder] = GenerateRecurrentInput;
            dict[nodeNames.PreviousActionPlaceholder] = GeneratePreviousActionInput;
            dict[nodeNames.ActionMaskPlaceholder] = GenerateActionMaskInput;
            dict[nodeNames.RandomNormalEpsilonPlaceholder] =
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
                    dict[nodeNames.VisualObservationPlaceholderPrefix + visIndex] =
                        (tensor, batchSize, agentInfo) =>
                            GenerateVisualObservationInput(tensor, agentInfo, index, bw);
                }
            }
        }
        
        public new Action<Tensor, int, Dictionary<Agent, AgentInfo>> this[string index]
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
        
        private static void GenerateBatchSize(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Data = new int[] {batchSize};
        }
        
        private static void GenerateSequenceLength(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo)
        {
            tensor.Data = new int[] {1};
        }
            
        private static void GenerateVectorObservation(
            Tensor tensor,
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
        
        private static void GenerateRecurrentInput(
            Tensor tensor,
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
        
        private static void GeneratePreviousActionInput(
            Tensor tensor,
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
        
        private static void GenerateActionMaskInput(
            Tensor tensor,
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

        private static void GenerateRandomNormalInput(
            Tensor tensor,
            int batchSize,
            Dictionary<Agent, AgentInfo> agentInfo,
            RandomNormal randomNormal)
        {
            tensor.Shape[0] = batchSize;
            var actionSize = tensor.Shape[1];
            tensor.Data = new float[batchSize, actionSize];
            randomNormal.FillTensor(tensor);
        }

        private static void GenerateVisualObservationInput(
            Tensor tensor,
            Dictionary<Agent, AgentInfo> agentInfo,
            int visIndex,
            bool bw)
        {
            //TODO : More efficient ?
            var textures = agentInfo.Keys.Select(
                agent => agentInfo[agent].visualObservations[visIndex]).ToList();
            tensor.Data = BatchVisualObservations(textures, bw);

        }
        
        /// <summary>
        /// Converts a list of Texture2D into a Tensor.
        /// </summary>
        /// <returns>
        /// A 4 dimensional float Tensor of dimension
        /// [batch_size, height, width, channel].
        /// Where batch_size is the number of input textures,
        /// height corresponds to the height of the texture,
        /// width corresponds to the width of the texture,
        /// channel corresponds to the number of channels extracted from the
        /// input textures (based on the input blackAndWhite flag
        /// (3 if the flag is false, 1 otherwise).
        /// The values of the Tensor are between 0 and 1.
        /// </returns>
        /// <param name="textures">
        /// The list of textures to be put into the tensor.
        /// Note that the textures must have same width and height.
        /// </param>
        /// <param name="blackAndWhite">
        /// If set to <c>true</c> the textures
        /// will be converted to grayscale before being stored in the tensor.
        /// </param>
        public static float[,,,] BatchVisualObservations(
            List<Texture2D> textures, bool blackAndWhite)
        {
            int batchSize = textures.Count();
            int width = textures[0].width;
            int height = textures[0].height;
            int pixels = 0;
            if (blackAndWhite)
                pixels = 1;
            else
                pixels = 3;
            float[,,,] result = new float[batchSize, height, width, pixels];
            float[] resultTemp = new float[batchSize * height * width * pixels];
            int hwp = height * width * pixels;
            int wp = width * pixels;

            for (int b = 0; b < batchSize; b++)
            {
                Color32[] cc = textures[b].GetPixels32();
                for (int h = height - 1; h >= 0; h--)
                {
                    for (int w = 0; w < width; w++)
                    {
                        Color32 currentPixel = cc[(height - h - 1) * width + w];
                        if (!blackAndWhite)
                        {
                            // For Color32, the r, g and b values are between
                            // 0 and 255.
                            resultTemp[b * hwp + h * wp + w * pixels] = currentPixel.r / 255.0f;
                            resultTemp[b * hwp + h * wp + w * pixels + 1] = currentPixel.g / 255.0f;
                            resultTemp[b * hwp + h * wp + w * pixels + 2] = currentPixel.b / 255.0f;
                        }
                        else
                        {
                            resultTemp[b * hwp + h * wp + w * pixels] =
                                (currentPixel.r + currentPixel.g + currentPixel.b)
                                / 3f / 255.0f;
                        }
                    }
                }
            }

            System.Buffer.BlockCopy(resultTemp, 0, result, 0, batchSize * hwp * sizeof(float));
            return result;
        }
        
    }
}