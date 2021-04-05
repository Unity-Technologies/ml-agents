using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class CountingGridSensorShapeTests
    {

        GameObject testGo;
        GridSensorComponent gridSensorComponent;

        [SetUp]
        public void SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensorComponent = testGo.AddComponent<GridSensorComponent>();
        }

        [TearDown]
        public void ClearScene()
        {
            Object.DestroyImmediate(testGo);
        }

        [Test]
        public void OneTagMoreDepthError()
        {
            string[] tags = { "block" };
            int[] depths = { 1, 1 };
            Color[] colors = { Color.magenta };
            GridObsTestUtils.SetComponentParameters(gridSensorComponent, tags, depths, GridDepthType.Counting, 1f, 1f, 10, 10, LayerMask.GetMask("Default"), false, colors);
            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensorComponent.CreateSensors();
            });
        }
    }
}
