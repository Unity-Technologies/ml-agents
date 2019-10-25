using UnityEngine;
using System.Collections.Generic;
using MLAgents.InferenceBrain;

namespace MLAgents
{
    public static class Utilities
    {
        /// <summary>
        /// Converts a list of Texture2D into a TensorProxy.
        /// </summary>
        /// <param name="textures">
        /// The list of textures to be put into the tensor.
        /// Note that the textures must have same width and height.
        /// </param>
        /// <param name="tensorProxy">
        /// TensorProxy to fill with Texture data.
        /// </param>
        /// <param name="grayScale">
        /// If set to <c>true</c> the textures will be converted to grayscale before
        /// being stored in the tensor.
        /// </param>
        public static void TextureToTensorProxy(
            List<Texture2D> textures,
            TensorProxy tensorProxy,
            bool grayScale)
        {
            var numTextures = textures.Count;
            var width = textures[0].width;
            var height = textures[0].height;

            for (var t = 0; t < numTextures; t++)
            {
                var texture = textures[t];
                Debug.Assert(width == texture.width, "All Textures must have the same dimension");
                Debug.Assert(height == texture.height, "All Textures must have the same dimension");
                TextureToTensorProxy(texture, tensorProxy, grayScale, t);
            }
        }

        /// <summary>
        /// Puts a Texture2D into a TensorProxy.
        /// </summary>
        /// <param name="texture">
        /// The texture to be put into the tensor.
        /// </param>
        /// <param name="tensorProxy">
        /// TensorProxy to fill with Texture data.
        /// </param>
        /// <param name="grayScale">
        /// If set to <c>true</c> the textures will be converted to grayscale before
        /// being stored in the tensor.
        /// </param>
        /// <param name="textureOffset">
        /// Index of the texture being written.
        /// </param>
        public static void TextureToTensorProxy(
            Texture2D texture,
            TensorProxy tensorProxy,
            bool grayScale,
            int textureOffset = 0)
        {
            var width = texture.width;
            var height = texture.height;
            var data = tensorProxy.data;

            var t = textureOffset;
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
                        data[t, h, w, 0] =
                            (currentPixel.r + currentPixel.g + currentPixel.b) / 3f / 255.0f;
                    }
                    else
                    {
                        // For Color32, the r, g and b values are between 0 and 255.
                        data[t, h, w, 0] = currentPixel.r / 255.0f;
                        data[t, h, w, 1] = currentPixel.g / 255.0f;
                        data[t, h, w, 2] = currentPixel.b / 255.0f;
                    }
                }
            }

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
    }
}
