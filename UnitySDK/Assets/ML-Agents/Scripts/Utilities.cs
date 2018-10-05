using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace MLAgents
{
    public class Utilities
    {
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
        public static float[,,,] TextureToFloatArray(
            List<Texture2D> textures, bool blackAndWhite)
        {
            int batchSize = textures.Count;
            int width = textures[0].width;
            int height = textures[0].height;
            var pixels = blackAndWhite ? 1 : 3;
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
        
        
        /// <summary>
        /// Calculates the cumulative sum of an integer array. The result array will be one element
        /// larger than the input array since it has a padded 0 at the begining.
        /// If the input is [a, b, c], the result will be [0, a, a+b, a+b+c]
        /// </summary>
        /// <returns> The cumulative sum of the input array.</returns>
        public static int[] CumSum(int [] array)
        {
            var runningSum = 0;
            var result = new int[array.Length + 1];
            for (var actionIndex = 0; actionIndex < array.Length; actionIndex++)
            {
                runningSum += array[actionIndex];
                result[actionIndex + 1] = runningSum;
            }
            return result;
        }
    }
}
