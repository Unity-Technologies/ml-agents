using System;

namespace Unity.MLAgents
{
    internal static class Utilities
    {

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
