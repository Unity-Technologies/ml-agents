using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.MLAgents.Extensions.Sensors;
using Unity.MLAgents.Extensions.TestUtils.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class ChannelPerceiveTests
    {
        GameObject testGo;
        GameObject boxGo;
        SimpleTestGridSensor gridSensor;
        GridSensorDummyData dummyData;

        // Use built-in tags
        const string k_Tag1 = "Player";

        [UnitySetUp]
        public IEnumerator SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensor = testGo.AddComponent<SimpleTestGridSensor>();

            boxGo = new GameObject("block");
            boxGo.tag = k_Tag1;
            boxGo.transform.position = new Vector3(3f, 0f, 3f);
            boxGo.AddComponent<BoxCollider>();
            dummyData = boxGo.AddComponent<GridSensorDummyData>();
            yield return null;
        }

        [TearDown]
        public void ClearScene()
        {
            Object.DestroyImmediate(boxGo);
            Object.DestroyImmediate(testGo);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthOneBelowZeroException()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1 };
            dummyData.Data = new[] { -0.1f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Perceive();
            });
        }

        [UnityTest]
        public IEnumerator OneChannelDepthOneAboveDepthException()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1 };
            dummyData.Data = new[] { 1.1f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Perceive();
            });
        }

        [UnityTest]
        public IEnumerator OneChannelDepthOneGoodValue()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1 };
            dummyData.Data = new[] { .2f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { .2f }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthThreeBelowZeroException()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 3 };
            dummyData.Data = new[] { -1f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Perceive();
            });
        }

        [UnityTest]
        public IEnumerator OneChannelDepthThreeAboveDepthException()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 3 };
            dummyData.Data = new[] { 4f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Perceive();
            });
        }

        [UnityTest]
        public IEnumerator OneChannelDepthThreeGoodValueInt()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 3 };
            dummyData.Data = new[] { 2f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 2f / 3f }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator OneChannelDepthThreeGoodValueFloat()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 3 };
            dummyData.Data = new[] { 2.4f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 2.4f / 3f }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthOneOneBelowZeroException()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1, 1 };
            dummyData.Data = new float[] { -1, 1 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Perceive();
            });
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthOneOneAboveDepthException()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1, 1 };
            dummyData.Data = new float[] { 1, 3 };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Perceive();
            });
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthOneOneGoodValues()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1, 1 };
            dummyData.Data = new[] { .4f, .3f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 2, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { .4f, .3f }, 4);
            float[] expectedDefault = new float[] { 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthOneThreeAboveDepthException()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1, 3 };
            dummyData.Data = new[] { .4f, 4f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Perceive();
            });
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthOneThreeGoodValues()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 1, 3 };
            dummyData.Data = new[] { .4f, 1f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 2, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { .4f, 1f / 3f }, 4);
            float[] expectedDefault = new float[] { 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthThreeOneGoodValues()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 3, 1 };
            dummyData.Data = new[] { 1f, .4f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 2, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 1f / 3f, .4f }, 4);
            float[] expectedDefault = new float[] { 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator TwoChannelDepthThreeThreeGoodValues()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 3, 3 };
            dummyData.Data = new[] { 1f, 2.2f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 2, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 1f / 3f, 2.2f / 3 }, 4);
            float[] expectedDefault = new float[] { 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator ThreeChannelDepthFiveOneThreeGoodValues()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 5, 1, 3 };
            dummyData.Data = new[] { 3f, .6f, 2.2f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 3, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 3f / 5f, .6f, 2.2f / 3f }, 4);
            float[] expectedDefault = new float[] { 0, 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator ProperReset()
        {
            string[] tags = { k_Tag1 };
            int[] depths = { 5, 1, 3 };
            dummyData.Data = new[] { 3f, .6f, 2.2f };
            Color[] colors = { Color.red, Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            yield return null;

            float[] output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 3, output.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 3f / 5f, .6f, 2.2f / 3f }, 4);
            float[] expectedDefault = new float[] { 0, 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);

            Object.DestroyImmediate(boxGo);

            yield return null;

            output = gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 3, output.Length);

            subarrayIndicies = new int[0];
            expectedSubarrays = new float[0][];
            GridObsTestUtils.AssertSubarraysAtIndex(output, subarrayIndicies, expectedSubarrays, expectedDefault);
        }
    }
}
