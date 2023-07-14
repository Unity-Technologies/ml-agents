using System;
using System.Collections;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Extensions.Sensors;
using Object = UnityEngine.Object;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class CountingGridSensorTests
    {
        GameObject testGo;
        GameObject boxGo;
        TestCountingGridSensorComponent gridSensorComponent;

        // Use built-in tags
        const string k_Tag1 = "Player";
        const string k_Tag2 = "Respawn";

        [UnitySetUp]
        public IEnumerator SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensorComponent = testGo.AddComponent<TestCountingGridSensorComponent>();

            boxGo = new GameObject("block");
            boxGo.tag = k_Tag1;
            boxGo.transform.position = new Vector3(3f, 0f, 3f);
            boxGo.AddComponent<BoxCollider>();

            yield return null;
        }

        [TearDown]
        public void ClearScene()
        {
            Object.DestroyImmediate(boxGo);
            Object.DestroyImmediate(testGo);
        }

        public class TestCountingGridSensorComponent : GridSensorComponent
        {
            public void SetParameters(string[] detectableTags)
            {
                DetectableTags = detectableTags;
                CellScale = new Vector3(1, 0.01f, 1);
                GridSize = new Vector3Int(10, 1, 10);
                ColliderMask = LayerMask.GetMask("Default");
                RotateWithAgent = false;
                CompressionType = SensorCompressionType.None;
            }

            protected override GridSensorBase[] GetGridSensors()
            {
                return new GridSensorBase[]
                {
                    new CountingGridSensor(
                        "TestSensor",
                        CellScale,
                        GridSize,
                        DetectableTags,
                        CompressionType)
                };
            }
        }

        // Copied from GridSensorTests in main package
        public static float[][] DuplicateArray(float[] array, int numCopies)
        {
            float[][] duplicated = new float[numCopies][];
            for (int i = 0; i < numCopies; i++)
            {
                duplicated[i] = array;
            }
            return duplicated;
        }

        // Copied from GridSensorTests in main package
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

        [Test]
        public void TestCountingSensor()
        {
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetParameters(tags);
            var gridSensor = (CountingGridSensor)gridSensorComponent.CreateSensors()[0];
            Assert.AreEqual(gridSensor.PerceptionBuffer.Length, 10 * 10 * 2);

            gridSensor.Update();

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = DuplicateArray(new float[] { 1, 0 }, 4);
            float[] expectedDefault = new float[] { 0, 0 };
            AssertSubarraysAtIndex(gridSensor.PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);

            var boxGo2 = new GameObject("block");
            boxGo2.tag = k_Tag1;
            boxGo2.transform.position = new Vector3(3.1f, 0f, 3f);
            boxGo2.AddComponent<BoxCollider>();

            gridSensor.Update();

            subarrayIndicies = new[] { 77, 78, 87, 88 };
            expectedSubarrays = DuplicateArray(new float[] { 2, 0 }, 4);
            expectedDefault = new float[] { 0, 0 };
            AssertSubarraysAtIndex(gridSensor.PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
            Object.DestroyImmediate(boxGo2);
        }
    }
}
