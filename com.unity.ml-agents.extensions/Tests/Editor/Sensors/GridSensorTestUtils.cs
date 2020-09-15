using NUnit.Framework;
using System;
using System.Diagnostics;
using System.Linq;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public static class GridObsTestUtils
    {


        /// <summary>
        /// Returns a human readable string of an array. Optional arguments are the index to start from and the number of elements to add to the string
        /// </summary>
        /// <param name="arr">The array to convert to string</param>
        /// <param name="initialIndex">The initial index. Default 0</param>
        /// <param name="numOfElements">The number of elements to print</param>
        /// <returns>Human readable string</returns>
        public static string Array2Str<T>(T[] arr, int initialIndex = 0, int maxNumberOfElements = int.MaxValue)
        {
            return String.Join(", ", arr.Skip(initialIndex).Take(maxNumberOfElements));
        }

        /// <summary>
        /// Given a flattened matrix and a shape, returns a string in human readable format
        /// </summary>
        /// <param name="arr">Flattened matrix array</param>
        /// <param name="shape">Shape of matrix</param>
        /// <returns>human readable string</returns>
        public static string Matrix2Str(float[] arr, int[] shape)
        {
            string log = "[";

            int t = 0;
            for (int i = 0; i < shape[0]; i++)
            {
                log += "\n[";
                for (int j = 0; j < shape[1]; j++)
                {
                    log += "[";
                    for (int k = 0; k < shape[2]; k++)
                    {
                        log += arr[t] + ", ";
                        t++;
                    }
                    log += "],";
                }
                log += "]";
            }
            log += "]";
            return log;
        }

        /// <summary>
        /// Utility function to duplicate an array into an array of arrays
        /// </summary>
        /// <param name="array">array to duplicate</param>
        /// <param name="numCopies">number of times to duplicate</param>
        /// <returns>array of duplicated arrays</returns>
        public static float[][] DuplicateArray(float[] array, int numCopies)
        {
            float[][] duplicated = new float[numCopies][];
            for (int i = 0; i < numCopies; i++)
            {
                duplicated[i] = array;
            }
            return duplicated;
        }

        /// <summary>
        /// Asserts that 2 int arrays are the same
        /// </summary>
        /// <param name="expected">The expected array</param>
        /// <param name="actual">The actual array</param>
        public static void AssertArraysAreEqual(int[] expected, int[] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length, "Lengths are not the same");
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], "Got " + Array2Str(actual) + ", expected " + Array2Str(expected));
            }
        }

        /// <summary>
        /// Asserts that 2 float arrays are the same
        /// </summary>
        /// <param name="expected">The expected array</param>
        /// <param name="actual">The actual array</param>
        public static void AssertArraysAreEqual(float[] expected, float[] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length, "Lengths are not the same");
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], "Got " + Array2Str(actual) + ", expected " + Array2Str(expected));
            }
        }

        /// <summary>
        /// Asserts that the sub-arrays of the total array are equal to specific subarrays at specific subarray indicies and equal to a default everywhere else.
        /// </summary>
        /// <param name="total">Array containing all data of the grid observation. Is a concatenation of N subarrays all of the same length</param>
        /// <param name="indicies">The indicies to verify that differ from the default array</param>
        /// <param name="expectedArrays">The sub arrays values that differ from the default array</param>
        /// <param name="expectedDefaultArray">The default value of a sub array</param>
        /// <example>
        /// If the total array is data from a 4x4x2 grid observation, total will be an array of size 32 and each sub array will have a size of 2.
        /// Let 3 cells at indicies (0, 1), (2, 2), and (3, 0) with values ([.1, .5]), ([.9, .7]), ([0, .2]), respectively.
        /// If the default values of cells are ([0, 0]) then the grid observation will be as follows:
        /// [ [0,  0], [.1, .5], [ 0, 0 ], [0, 0], 
        ///   [0,  0], [ 0, 0 ], [ 0, 0 ], [0, 0],
        ///   [0,  0], [ 0, 0 ], [.9, .7], [0, 0],
        ///   [0, .2], [ 0, 0 ], [ 0, 0 ],  [0, 0] ]
        ///   
        /// Which will make the total array will be the flattened array
        /// total = [0, 0, .1, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .9, .7, 0, 0, 0, .2, 0, 0, 0, 0, 0]
        /// 
        /// The indicies of the activated cells in the flattened array will be 1, 10, and 12
        ///
        /// So to verify that the total array is as expected, AssertSubarraysAtIndex should be called as
        /// AssertSubarraysAtIndex(
        ///             total,
        ///             indicies = new int[] {1, 10, 12},
        ///             expectedArrays = new float[][] { new float[] {.1, .5}, new float[] {.9, .7}, new float[] {0, .2}},
        ///             expecedDefaultArray = new float[] {0, 0}
        ///             )
        /// </example>
        public static void AssertSubarraysAtIndex(float[] total, int[] indicies, float[][] expectedArrays, float[] expectedDefaultArray)
        {
            int totalIndex = 0;
            int subIndex = 0;
            int subarrayIndex = 0;
            int lenOfData = expectedDefaultArray.Length;
            int numArrays = total.Length / lenOfData;
            for (int i = 0; i < numArrays; i++)
            {
                totalIndex = i * lenOfData;

                if (indicies.Contains(i))
                {
                    subarrayIndex = Array.IndexOf(indicies, i);
                    for (subIndex = 0; subIndex < lenOfData; subIndex++)
                    {
                        Assert.AreEqual(expectedArrays[subarrayIndex][subIndex], total[totalIndex],
                            "Expected " + expectedArrays[subarrayIndex][subIndex] + " at subarray index " + totalIndex + ", index = " + subIndex + " but was " + total[totalIndex]);
                        totalIndex++;
                    }
                }
                else
                {
                    for (subIndex = 0; subIndex < lenOfData; subIndex++)
                    {
                        Assert.AreEqual(expectedDefaultArray[subIndex], total[totalIndex],
                            "Expected default value " + expectedDefaultArray[subIndex] + " at subarray index " + totalIndex + ", index = " + subIndex + " but was " + total[totalIndex]);
                        totalIndex++;
                    }
                }
            }
        }

    }
}
