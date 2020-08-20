using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class CountingGridSensorPerceiveTests
    {
        GameObject testGo;
        GameObject boxGo;
        GameObject boxGoTwo;
        GameObject boxGoThree;
        CountingGridSensor gridSensor;

        public GameObject CreateBlock(Vector3 postion, string tag, string name)
        {
            GameObject boxGo = new GameObject(name);
            boxGo.tag = tag;
            boxGo.transform.position = postion;
            boxGo.AddComponent<BoxCollider>();
            return boxGo;
        }

        [UnitySetUp]
        public IEnumerator SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensor = testGo.AddComponent<CountingGridSensor>();

            boxGo = CreateBlock(new Vector3(3f, 0f, 3f), "block", "box1");

            yield return null;
        }

        [TearDown]
        public void ClearScene()
        {
            Object.DestroyImmediate(boxGo);
            Object.DestroyImmediate(boxGoTwo);
            Object.DestroyImmediate(boxGoThree);
            Object.DestroyImmediate(testGo);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthOneCount()
        {
            string[] tags = { "block" };
            int[] depths = { 1 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { 1f }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthOneCountMax()
        {

            string[] tags = { "block" };
            int[] depths = { 1 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            boxGoTwo = CreateBlock(new Vector3(3.1f, 0f, 3.1f), "block", "box2");

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { 1f }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthFourCount()
        {

            string[] tags = { "block" };
            int[] depths = { 4 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            boxGoTwo = CreateBlock(new Vector3(3.1f, 0f, 3.1f), "block", "box2");

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { .5f }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthFourCount()
        {

            string[] tags = { "block", "food" };
            int[] depths = { 4, 1 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            boxGoTwo = CreateBlock(new Vector3(3.1f, 0f, 3.1f), "block", "box2");
            boxGoThree = CreateBlock(new Vector3(2.9f, 0f, 2.9f), "food", "box2");

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 2, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { .5f, 1 }, 4);
            float[] expectedDefault = new float[] { 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }
    }
}
