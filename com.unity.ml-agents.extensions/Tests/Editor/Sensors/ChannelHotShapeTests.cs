using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Extensions.Sensors;
using NUnit.Framework.Internal;
using UnityEngine.TestTools;
using System.Collections;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class ChannelHotShapeTests
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
        public void OneChannelDepthOne()
        {
            string[] tags = { "Box", "Ball" };
            int[] depths = { 1 };
            Color[] colors = { Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.ChannelHot,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            int[] expectedShape = { 10, 10, 1 };
            GridObsTestUtils.AssertArraysAreEqual(expectedShape, gridSensor.GetFloatObservationShape());

        }


        [Test]
        public void OneChannelDepthTwo()
        {

            string[] tags = { "Box", "Ball" };
            int[] depths = { 2 };
            Color[] colors = { Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.ChannelHot,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            int[] expectedShape = { 10, 10, 2 };
            GridObsTestUtils.AssertArraysAreEqual(expectedShape, gridSensor.GetFloatObservationShape());

        }

        [Test]
        public void TwoChannelsDepthTwoOne()
        {
            string[] tags = { "Box", "Ball" };
            int[] depths = { 2, 1 };
            Color[] colors = { Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.ChannelHot,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            int[] expectedShape = { 10, 10, 3 };
            GridObsTestUtils.AssertArraysAreEqual(expectedShape, gridSensor.GetFloatObservationShape());

        }

        [Test]
        public void TwoChannelsDepthThreeThree()
        {
            string[] tags = { "Box", "Ball" };
            int[] depths = { 3, 3 };
            Color[] colors = { Color.magenta };
            gridSensor.SetParameters(tags, depths, GridSensor.GridDepthType.ChannelHot,
                1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            gridSensor.Start();

            int[] expectedShape = { 10, 10, 6 };
            GridObsTestUtils.AssertArraysAreEqual(expectedShape, gridSensor.GetFloatObservationShape());

        }

    }
}
