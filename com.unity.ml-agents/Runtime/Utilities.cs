using UnityEngine;
using System.Collections.Generic;

namespace MLAgents
{
    internal static class Utilities
    {
        /// <summary>
        /// Puts a Texture2D into a WriteAdapter.
        /// </summary>
        /// <param name="texture">
        /// The texture to be put into the tensor.
        /// </param>
        /// <param name="adapter">
        /// Adapter to fill with Texture data.
        /// </param>
        /// <param name="grayScale">
        /// If set to <c>true</c> the textures will be converted to grayscale before
        /// being stored in the tensor.
        /// </param>
        /// <returns>The number of floats written</returns>
        public static int TextureToTensorProxy(
            Texture2D texture,
            WriteAdapter adapter,
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
                        adapter[h, w, 0] =
                            (currentPixel.r + currentPixel.g + currentPixel.b) / 3f / 255.0f;
                    }
                    else
                    {
                        // For Color32, the r, g and b values are between 0 and 255.
                        adapter[h, w, 0] = currentPixel.r / 255.0f;
                        adapter[h, w, 1] = currentPixel.g / 255.0f;
                        adapter[h, w, 2] = currentPixel.b / 255.0f;
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
        public static int[] CumSum(int[] input)
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

        /// <summary>
        /// Shifts list elements to the left by the specified amount (in-place).
        /// <param name="list">
        /// List whose elements will be shifted
        /// </param>
        /// <param name="shiftAmount">
        /// Amount to shift the elements to the left by
        /// </param>
        /// </summary>
        public static void ShiftLeft<T>(List<T> list, int shiftAmount)
        {
            for (var i = shiftAmount; i < list.Count; i++)
            {
                list[i - shiftAmount] = list[i];
            }
        }

        /// <summary>
        /// Replaces target list elements with source list elements starting at specified position
        /// in target list.
        /// <param name="dst">
        /// Target list, where the elements are added to
        /// </param>
        /// <param name="src">
        /// Source array, where the elements are copied from
        /// </param>
        /// <param name="start">
        /// Starting position in target list to copy elements to
        /// </param>
        /// </summary>
        public static void ReplaceRange<T>(List<T> dst, List<T> src, int start)
        {
            for (var i = 0; i < src.Count; i++)
            {
                dst[i + start] = src[i];
            }
        }

        /// <summary>
        /// Adds elements to list without extra temp allocations (assuming it fits pre-allocated
        /// capacity of the list). The built-in List/<T/>.AddRange() unfortunately allocates
        /// a temporary list to add items (even if the original array has sufficient capacity):
        /// https://stackoverflow.com/questions/2123161/listt-addrange-implementation-suboptimal
        /// Note: this implementation might be slow with a large source array.
        /// <param name="dst">
        /// Target list, where the elements are added to
        /// </param>
        /// <param name="src">
        /// Source array, where the elements are copied from
        /// </param>
        /// </summary>
        // ReSharper disable once ParameterTypeCanBeEnumerable.Global
        public static void AddRangeNoAlloc<T>(List<T> dst, T[] src)
        {
            // ReSharper disable once LoopCanBeConvertedToQuery
            foreach (var item in src)
            {
                dst.Add(item);
            }
        }

        /// <summary>
        /// Calculates the number of uncompressed floats in a list of ISensor
        /// </summary>
        public static int GetSensorFloatObservationSize(this List<ISensor> sensors)
        {
            int numFloatObservations = 0;
            for (var i = 0; i < sensors.Count; i++)
            {
                if (sensors[i].GetCompressionType() == SensorCompressionType.None)
                {
                    numFloatObservations += sensors[i].ObservationSize();
                }
            }
            return numFloatObservations;
        }
    }
}
