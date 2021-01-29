using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class ChannelShapeTests
    {

        GameObject testGo;
        GridSensor gridSensor;

        [SetUp]
        public void SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensor = testGo.AddComponent<GridSensor>();
        }

        [TearDown]
        public void ClearScene()
        {
            Object.DestroyImmediate(testGo);
        }

        [Test]
        public void OneChannel()
        {
            string[] tags = { "Box" };
            int[] depths = { 1 };
            Color[] colors = { Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            int[] expectedShape = { 10, 10, 1 };
            GridObsTestUtils.AssertArraysAreEqual(expectedShape, gridSensor.GetFloatObservationShape());
        }

        [Test]
        public void TwoChannel()
        {
            string[] tags = { "Box", "Ball" };
            int[] depths = { 1, 1 };
            Color[] colors = { Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            int[] expectedShape = { 10, 10, 2 };
            GridObsTestUtils.AssertArraysAreEqual(expectedShape, gridSensor.GetFloatObservationShape());
        }

        [Test]
        public void SevenChannel()
        {
            string[] tags = { "Box", "Ball" };
            int[] depths = { 1, 1, 1, 1, 1, 1, 1 };
            Color[] colors = { Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.Channel,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            int[] expectedShape = { 10, 10, 7 };
            GridObsTestUtils.AssertArraysAreEqual(expectedShape, gridSensor.GetFloatObservationShape());
        }
    }
}
