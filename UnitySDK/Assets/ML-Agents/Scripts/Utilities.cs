using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Barracuda;
using MLAgents.InferenceBrain;

namespace MLAgents
{
    public class Utilities
    {
        /// <summary>
        /// Converts a list of Texture2D into a Tensor.
        /// </summary>
        /// <param name="tensorProxy">
        /// Tensor proxy to fill with Texture data.
        /// </param>
        /// <param name="textures">
        /// The list of textures to be put into the tensor.
        /// Note that the textures must have same width and height.
        /// </param>
        /// <param name="blackAndWhite">
        /// If set to <c>true</c> the textures
        /// will be converted to grayscale before being stored in the tensor.
        /// </param>
        /// <param name="allocator">Tensor allocator</param>
        public static void TextureToTensorProxy(TensorProxy tensorProxy, List<Texture2D> textures, bool blackAndWhite, 
                                                                ITensorAllocator allocator)
        {
            var batchSize = textures.Count;
            var width = textures[0].width;
            var height = textures[0].height;
            var data = tensorProxy.Data;

            for (var b = 0; b < batchSize; b++)
            {
                var cc = textures[b].GetPixels32();
                for (var h = height - 1; h >= 0; h--)
                {
                    for (var w = 0; w < width; w++)
                    {
                        var currentPixel = cc[(height - h - 1) * width + w];
                        if (!blackAndWhite)
                        {
                            // For Color32, the r, g and b values are between
                            // 0 and 255.
                            data[b, h, w, 0] = currentPixel.r / 255.0f;
                            data[b, h, w, 1] = currentPixel.g / 255.0f;
                            data[b, h, w,2] = currentPixel.b / 255.0f;
                        }
                        else
                        {
                            data[b, h, w, 0] = (currentPixel.r + currentPixel.g + currentPixel.b)
                                / 3f / 255.0f;
                        }
                    }
                }
            }
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

        /// <summary>
        /// Shifts list elements to the left by the specified amount.
        /// <param name="list">
        /// Target list
        /// </param>
        /// <param name="amount">
        /// Shift amount
        /// </param>
        /// </summary>
        public static void ShiftLeft<T>(List<T> list, int amount)
        {
            for (var i = amount; i < list.Count; i++)
            {
                list[i - amount] = list[i];
            }
        }

        /// <summary>
        /// Replaces target list elements with source list elements starting at specified position in target list.
        /// <param name="dst">
        /// Target list
        /// </param>
        /// <param name="src">
        /// Source list
        /// </param>
        /// <param name="start">
        /// Offset in target list
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
        /// Adds elements to list without extra temp allocations (assuming it fits pre-allocated capacity of the list).
        /// Regular List<T>.AddRange() unfortunately allocates temp list to add items.
        /// https://stackoverflow.com/questions/2123161/listt-addrange-implementation-suboptimal
        /// Note: this implementation might be slow with large numbers of elements in the source array.
        /// <param name="dst">
        /// Target list
        /// </param>
        /// <param name="src">
        /// Source array
        /// </param>
        /// </summary>
        public static void AddRangeNoAlloc<T>(List<T> dst, T[] src)
        {
            var offset = dst.Count;
            
            for (var i = 0; i < src.Length; i++)
            {
                dst.Add(src[i]);
            }
        }
    }
}
