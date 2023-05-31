#if MLA_UNITY_PHYSICS_MODULE
using System.Collections.Generic;
using System.Collections;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    public class GridSensorTests
    {
        GameObject testGo;
        GameObject boxGo;
        SimpleTestGridSensorComponent gridSensorComponent;

        // Use built-in tags
        const string k_Tag1 = "Player";
        const string k_Tag2 = "Respawn";

        [UnitySetUp]
        public IEnumerator SetupScene()
        {
            testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            gridSensorComponent = testGo.AddComponent<SimpleTestGridSensorComponent>();

            boxGo = new GameObject("block");
            boxGo.tag = k_Tag1;
            boxGo.transform.position = new Vector3(3f, 0f, 3f);
            boxGo.AddComponent<BoxCollider>();

            TestGridSensorConfig.Reset();
            yield return null;
        }

        [TearDown]
        public void ClearScene()
        {
            Object.DestroyImmediate(boxGo);
            Object.DestroyImmediate(testGo);
        }

        [Test]
        public void TestBufferSize()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, gridSizeX: 3, gridSizeZ: 4, useTestingGridSensor: true);
            TestGridSensorConfig.SetParameters(5, true, false);
            var gridSensor = (SimpleTestGridSensor)gridSensorComponent.CreateSensors()[0];
            Assert.AreEqual(gridSensor.PerceptionBuffer.Length, 3 * 4 * 5);
        }

        [Test]
        public void TestInvalidSizeConfiguration()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, gridSizeY: 10, useTestingGridSensor: true);
            gridSensorComponent.CreateSensors(); // expect no exception

            gridSensorComponent.m_GridSize.y = 10;
            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensorComponent.CreateSensors();
            });
        }

        [Test]
        public void TestInvalidCompressionConfiguration()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, compression: SensorCompressionType.PNG, useTestingGridSensor: true);

            var gridSensor = (GridSensorBase)gridSensorComponent.CreateSensors()[0];
            LogAssert.Expect(LogType.Warning, $"Compression type {SensorCompressionType.PNG} is only supported with normalized data. " +
                        "The sensor will not compress the data.");
            Assert.AreEqual(gridSensor.CompressionType, SensorCompressionType.None);
        }

        [Test]
        public void TestCreateSensor()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, useGridSensorBase: true);

            gridSensorComponent.CreateSensors();
            var componentSensor = (List<GridSensorBase>)typeof(GridSensorComponent).GetField("m_Sensors",
                        BindingFlags.Instance | BindingFlags.NonPublic).GetValue(gridSensorComponent);
            Assert.AreEqual(componentSensor.Count, 1);
        }

        [Test]
        public void PerceiveNotSelf()
        {
            testGo.tag = k_Tag2;

            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, useGridSensorBase: true);
            var gridSensor = (GridSensorBase)gridSensorComponent.CreateSensors()[0];

            gridSensor.Update();

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { 1 }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(gridSensor.PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [Test]
        public void TestReset()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, useGridSensorBase: true);
            TestGridSensorConfig.SetParameters(3, false, false);
            var gridSensor = (GridSensorBase)gridSensorComponent.CreateSensors()[0];

            gridSensor.Update();

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { 1 }, 4);
            float[] expectedDefault = new float[] { 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(gridSensor.PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
            Object.DestroyImmediate(boxGo);

            gridSensor.Update();

            subarrayIndicies = new int[0];
            expectedSubarrays = new float[0][];
            GridObsTestUtils.AssertSubarraysAtIndex(gridSensor.PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [Test]
        public void TestOneHotSensor()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, useOneHotTag: true);
            var gridSensor = (OneHotGridSensor)gridSensorComponent.CreateSensors()[0];
            Assert.AreEqual(gridSensor.PerceptionBuffer.Length, 10 * 10 * 2);

            gridSensor.Update();

            int[] subarrayIndicies = new int[] { 77, 78, 87, 88 };
            float[][] expectedSubarrays = GridObsTestUtils.DuplicateArray(new float[] { 1, 0 }, 4);
            float[] expectedDefault = new float[] { 0, 0 };
            GridObsTestUtils.AssertSubarraysAtIndex(gridSensor.PerceptionBuffer, subarrayIndicies, expectedSubarrays, expectedDefault);
        }

        [Test]
        public void TestCustomSensorInvalidData()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, compression: SensorCompressionType.PNG, useTestingGridSensor: true);
            TestGridSensorConfig.SetParameters(5, true, false);
            var gridSensor = (SimpleTestGridSensor)gridSensorComponent.CreateSensors()[0];

            gridSensor.DummyData = new float[] { 1, 2, 3, 4, 5 };
            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensor.Update();
            });
        }

        [Test]
        public void TestMultipleSensors()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, useOneHotTag: true, useGridSensorBase: true, useTestingGridSensor: true);
            var gridSensors = gridSensorComponent.CreateSensors();
            Assert.IsNotNull(((GridSensorBase)gridSensors[0]).m_GridPerception);
            Assert.IsNull(((GridSensorBase)gridSensors[1]).m_GridPerception);
            Assert.IsNull(((GridSensorBase)gridSensors[2]).m_GridPerception);
        }

        [Test]
        public void TestNoSensors()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags);
            Assert.Throws<UnityAgentsException>(() =>
            {
                gridSensorComponent.CreateSensors();
            });
        }

        [Test]
        public void TestStackedSensors()
        {
            testGo.tag = k_Tag2;
            string[] tags = { k_Tag1, k_Tag2 };
            gridSensorComponent.SetComponentParameters(tags, useGridSensorBase: true);
            gridSensorComponent.ObservationStacks = 3;
            var sensors = gridSensorComponent.CreateSensors();
            Assert.IsInstanceOf(typeof(StackingSensor), sensors[0]);
        }
    }
}
#endif
