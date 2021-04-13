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
        GridSensorComponent gridSensorComponent;

        // Use built-in tags
        const string k_Tag1 = "Player";
        const string k_Tag2 = "Respawn";

        [UnitySetUp]
        public IEnumerator SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensorComponent = testGo.AddComponent<GridSensorComponent>();

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
            GridObsTestUtils.SetComponentParameters(gridSensorComponent, tags, depths, GridDepthType.ChannelHot,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            var gridSensor = (GridSensor)gridSensorComponent.CreateSensors()[0];

            yield return null;

            gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 3, gridSensor.m_PerceptionBuffer.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { 0, 1, 0 }, 4);
            float[] expectedDefault = new float[] { 0, 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(gridSensor.m_PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator PerceiveNotSelfChannel()
        {
            testGo.tag = k_Tag1;

            string[] tags = { k_Tag2, k_Tag1 };
            int[] depths = { 3 };
            Color[] colors = { Color.red, Color.magenta };
            GridObsTestUtils.SetComponentParameters(gridSensorComponent, tags, depths, GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            var gridSensor = (GridSensor)gridSensorComponent.CreateSensors()[0];

            yield return null;

            gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, gridSensor.m_PerceptionBuffer.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 1f / 3f }, 4);
            float[] expectedDefault = new float[] { 0f };
            GridObsTestUtils.AssertSubarraysAtIndex(gridSensor.m_PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [UnityTest]
        public IEnumerator PerceiveNotSelfChannelCount()
        {
            testGo.tag = k_Tag1;

            string[] tags = { k_Tag1 };
            int[] depths = { 3 };
            Color[] colors = { Color.red, Color.magenta };
            GridObsTestUtils.SetComponentParameters(gridSensorComponent, tags, depths, GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            var gridSensor = (GridSensor)gridSensorComponent.CreateSensors()[0];

            yield return null;

            gridSensor.Perceive();

            Assert.AreEqual(10 * 10 * 1, gridSensor.m_PerceptionBuffer.Length);

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new[] { 0f }, 4);
            float[] expectedDefault = new float[] { 0f };
            GridObsTestUtils.AssertSubarraysAtIndex(gridSensor.m_PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
        }
    }
}
