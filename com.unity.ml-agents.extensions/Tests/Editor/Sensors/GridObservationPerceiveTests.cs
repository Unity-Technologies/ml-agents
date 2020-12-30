using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class GridObservationPerceiveTests
    {
        GameObject testGo;
        GameObject boxGo;
        GridSensor gridSensor;

        // Use built-in tags
        const string k_Tag1 = "Player";
        const string k_Tag2 = "Respawn";

        [UnitySetUp]
        public IEnumerator SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensor = testGo.AddComponent<GridSensor>();

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

        [UnityTest]
        public IEnumerator PerceiveNotSelfChannelHot()
        {
            testGo.tag = k_Tag1;

            string[] tags = { k_Tag2, k_Tag1 };
            int[] depths = { 3 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.ChannelHot,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 3, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { 0, 0, 1 }, 4);
            float[] expectedDefault = new float[] { 1, 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator PerceiveNotSelfChannel()
        {
            testGo.tag = k_Tag1;

            string[] tags = { k_Tag2, k_Tag1 };
            int[] depths = { 3 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 2f / 3f }, 4);
            float[] expectedDefault = new float[] { 0f };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator PerceiveNotSelfChannelCount()
        {
            testGo.tag = k_Tag1;

            string[] tags = { k_Tag1 };
            int[] depths = { 3 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 1f / 3f }, 4);
            float[] expectedDefault = new float[] { 0f };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }
    }
}
