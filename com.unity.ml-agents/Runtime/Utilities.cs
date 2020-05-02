using System;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents
{
    internal static class Utilities
    {
        /// <summary>
        /// Puts a Texture2D into a ObservationWriter.
        /// </summary>
        /// <param name="texture">
        /// The texture to be put into the tensor.
        /// </param>
        /// <param name="obsWriter">
        /// Writer to fill with Texture data.
        /// </param>
        /// <param name="grayScale">
        /// If set to <c>true</c> the textures will be converted to grayscale before
        /// being stored in the tensor.
        /// </param>
        /// <returns>The number of floats written</returns>
        internal static int TextureToTensorProxy(
            Texture2D texture,
            ObservationWriter obsWriter,
            bool grayScale)
        {
            var width = texture.width;
            var height = texture.height;

            var texturePixels = texture.GetPixels32();
            // During training, we convert from Texture to PNG before sending to the trainer, which has the
            // effect of flipping the image. We need another flip here at inference time to match this.
            for (var h = height - 1; h >= 0; h--)
            {
                for (var w = 0; w < width; w++)
                {
                    var currentPixel = texturePixels[(height - h - 1) * width + w];
                    if (grayScale)
                    {
                        obsWriter[h, w, 0] =
                            (currentPixel.r + currentPixel.g + currentPixel.b) / 3f / 255.0f;
                    }
                    else
                    {
                        // For Color32, the r, g and b values are between 0 and 255.
                        obsWriter[h, w, 0] = currentPixel.r / 255.0f;
                        obsWriter[h, w, 1] = currentPixel.g / 255.0f;
                        obsWriter[h, w, 2] = currentPixel.b / 255.0f;
                    }
                }
            }

            return height * width * (grayScale ? 1 : 3);
        }

        /// <summary>
        /// Calculates the cumulative sum of an integer array. The result array will be one element
        /// larger than the input array since it has a padded 0 at the beginning.
        /// If the input is [a, b, c], the result will be [0, a, a+b, a+b+c]
        /// </summary>
        /// <param name="input">
        /// Input array whose elements will be cumulatively added
        /// </param>
        /// <returns> The cumulative sum of the input array.</returns>
        internal static int[] CumSum(int[] input)
        {
            var runningSum = 0;
            var result = new int[input.Length + 1];
            for (var actionIndex = 0; actionIndex < input.Length; actionIndex++)
            {
                runningSum += input[actionIndex];
                result[actionIndex + 1] = runningSum;
            }
            return result;
        }

#if DEBUG
        internal static void DebugCheckNanAndInfinity(float value, string valueCategory, string caller)
        {

            if (float.IsNaN(value))
            {
                throw new ArgumentException($"NaN {valueCategory} passed to {caller}.");
            }
            if (float.IsInfinity(value))
            {
                throw new ArgumentException($"Inifinity {valueCategory} passed to {caller}.");
            }
        }
#endif
    }

}
