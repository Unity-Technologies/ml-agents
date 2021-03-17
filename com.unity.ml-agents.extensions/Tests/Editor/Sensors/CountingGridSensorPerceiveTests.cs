using System.Collections;
using NUnit.Framework;
using Unity.Collections;
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

        // Use built-in tags
        const string k_Tag1 = "Player";
        const string k_Tag2 = "Respawn";


        public GameObject CreateBlock(Vector3 postion, string tag, string name)
        {
            var boxGo = new GameObject(name);
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

            boxGo = CreateBlock(new Vector3(3f, 0f, 3f), k_Tag1, "box1");

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
            string[] tags = { k_Tag1 };
            int[] depths = { 1 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            var output = gridSensor.Perceive(); gridSensor.UpdateBufferFromJob();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            var subarrayIndicies = new int[] { 77, 78, 87, 88 };
            var expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 1f }, 4);
            var expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthOneCountMax()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            boxGoTwo = CreateBlock(new Vector3(3.1f, 0f, 3.1f), k_Tag1, "box2");

            yield return null;

            var output = gridSensor.Perceive(); gridSensor.UpdateBufferFromJob();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            var subarrayIndicies = new int[] { 77, 78, 87, 88 };
            var expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 1f }, 4);
            var expectedDefault = new float[] { 0f };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthFourCount()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 4 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            boxGoTwo = CreateBlock(new Vector3(3.1f, 0f, 3.1f), k_Tag1, "box2");

            yield return null;

            var output = gridSensor.Perceive();
            gridSensor.UpdateBufferFromJob();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            var subarrayIndicies = new int[] { 77, 78, 87, 88 };
            var expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { .25f }, 4);
            var expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthFourCount()
        {
            string[] tags = { k_Tag1, k_Tag2 };
            int[] depths = { 4, 1 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            boxGoTwo = CreateBlock(new Vector3(3.1f, 0f, 3.1f), k_Tag1, "box2");
            boxGoThree = CreateBlock(new Vector3(2.9f, 0f, 2.9f), k_Tag2, "box2");

            yield return null;

            var output = gridSensor.Perceive();
            gridSensor.UpdateBufferFromJob();

            Assert.AreEqual(10 * 10 * 2, output.Length);

            var subarrayIndicies = new int[] { 77, 78, 87, 88 };
            var expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 0f, 1 }, 4);
            var expectedDefault = new float[] { 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }
    }
}
