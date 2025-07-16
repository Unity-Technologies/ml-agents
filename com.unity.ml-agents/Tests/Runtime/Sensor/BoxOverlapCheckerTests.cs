#if MLA_UNITY_PHYSICS_MODULE
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    internal class TestBoxOverlapChecker : BoxOverlapChecker
    {
        public TestBoxOverlapChecker(
            Vector3 cellScale,
            Vector3Int gridSize,
            bool rotateWithAgent,
            LayerMask colliderMask,
            GameObject centerObject,
            GameObject agentGameObject,
            string[] detectableTags,
            int initialColliderBufferSize,
            int maxColliderBufferSize
        ) : base(
            cellScale,
            gridSize,
            rotateWithAgent,
            colliderMask,
            centerObject,
            agentGameObject,
            detectableTags,
            initialColliderBufferSize,
            maxColliderBufferSize)
        {}

        public Vector3[] CellLocalPositions
        {
            get
            {
                return (Vector3[])typeof(BoxOverlapChecker).GetField("m_CellLocalPositions",
                    BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
            }
        }

        public Collider[] ColliderBuffer
        {
            get
            {
                return (Collider[])typeof(BoxOverlapChecker).GetField("m_ColliderBuffer",
                    BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
            }
        }

        public static TestBoxOverlapChecker CreateChecker(
            float cellScaleX = 1f,
            float cellScaleZ = 1f,
            int gridSizeX = 10,
            int gridSizeZ = 10,
            bool rotateWithAgent = true,
            GameObject centerObject = null,
            GameObject agentGameObject = null,
            string[] detectableTags = null,
            int initialColliderBufferSize = 4,
            int maxColliderBufferSize = 500)
        {
            return new TestBoxOverlapChecker(
                new Vector3(cellScaleX, 0.01f, cellScaleZ),
                new Vector3Int(gridSizeX, 1, gridSizeZ),
                rotateWithAgent,
                LayerMask.GetMask("Default"),
                centerObject,
                agentGameObject,
                detectableTags,
                initialColliderBufferSize,
                maxColliderBufferSize);
        }
    }

    public class BoxOverlapCheckerTests
    {
        [Test]
        public void TestCellLocalPosition()
        {
            var testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            var boxOverlapSquare = TestBoxOverlapChecker.CreateChecker(gridSizeX: 10, gridSizeZ: 10, rotateWithAgent: false, agentGameObject: testGo);

            var localPos = boxOverlapSquare.CellLocalPositions;
            Assert.AreEqual(new Vector3(-4.5f, 0, -4.5f), localPos[0]);
            Assert.AreEqual(new Vector3(-4.5f, 0, 4.5f), localPos[9]);
            Assert.AreEqual(new Vector3(4.5f, 0, -4.5f), localPos[90]);
            Assert.AreEqual(new Vector3(4.5f, 0, 4.5f), localPos[99]);
            Object.DestroyImmediate(testGo);

            var testGo2 = new GameObject("test");
            testGo2.transform.position = new Vector3(3.5f, 8f, 17f); // random, should have no effect on local positions
            var boxOverlapRect = TestBoxOverlapChecker.CreateChecker(gridSizeX: 5, gridSizeZ: 15, rotateWithAgent: true, agentGameObject: testGo);

            localPos = boxOverlapRect.CellLocalPositions;
            Assert.AreEqual(new Vector3(-2f, 0, -7f), localPos[0]);
            Assert.AreEqual(new Vector3(-2f, 0, 7f), localPos[14]);
            Assert.AreEqual(new Vector3(2f, 0, -7f), localPos[60]);
            Assert.AreEqual(new Vector3(2f, 0, 7f), localPos[74]);
            Object.DestroyImmediate(testGo2);
        }

        [Test]
        public void TestCellGlobalPositionNoRotate()
        {
            var testGo = new GameObject("test");
            var position = new Vector3(3.5f, 8f, 17f);
            testGo.transform.position = position;
            var boxOverlap = TestBoxOverlapChecker.CreateChecker(gridSizeX: 10, gridSizeZ: 10, rotateWithAgent: false, agentGameObject: testGo, centerObject: testGo);

            Assert.AreEqual(new Vector3(-4.5f, 0, -4.5f) + position, boxOverlap.GetCellGlobalPosition(0));
            Assert.AreEqual(new Vector3(-4.5f, 0, 4.5f) + position, boxOverlap.GetCellGlobalPosition(9));
            Assert.AreEqual(new Vector3(4.5f, 0, -4.5f) + position, boxOverlap.GetCellGlobalPosition(90));
            Assert.AreEqual(new Vector3(4.5f, 0, 4.5f) + position, boxOverlap.GetCellGlobalPosition(99));

            testGo.transform.Rotate(0, 90, 0); // should have no effect on positions
            Assert.AreEqual(new Vector3(-4.5f, 0, -4.5f) + position, boxOverlap.GetCellGlobalPosition(0));
            Assert.AreEqual(new Vector3(-4.5f, 0, 4.5f) + position, boxOverlap.GetCellGlobalPosition(9));
            Assert.AreEqual(new Vector3(4.5f, 0, -4.5f) + position, boxOverlap.GetCellGlobalPosition(90));
            Assert.AreEqual(new Vector3(4.5f, 0, 4.5f) + position, boxOverlap.GetCellGlobalPosition(99));

            Object.DestroyImmediate(testGo);
        }

        [Test]
        public void TestCellGlobalPositionRotate()
        {
            var testGo = new GameObject("test");
            var position = new Vector3(15f, 6f, 13f);
            testGo.transform.position = position;
            var boxOverlap = TestBoxOverlapChecker.CreateChecker(gridSizeX: 5, gridSizeZ: 15, rotateWithAgent: true, agentGameObject: testGo, centerObject: testGo);

            Assert.AreEqual(new Vector3(-2f, 0, -7f) + position, boxOverlap.GetCellGlobalPosition(0));
            Assert.AreEqual(new Vector3(-2f, 0, 7f) + position, boxOverlap.GetCellGlobalPosition(14));
            Assert.AreEqual(new Vector3(2f, 0, -7f) + position, boxOverlap.GetCellGlobalPosition(60));
            Assert.AreEqual(new Vector3(2f, 0, 7f) + position, boxOverlap.GetCellGlobalPosition(74));

            testGo.transform.Rotate(0, 90, 0);
            // round to int to ignore numeric errors
            Assert.AreEqual(Vector3Int.RoundToInt(new Vector3(-7f, 0, 2f) + position), Vector3Int.RoundToInt(boxOverlap.GetCellGlobalPosition(0)));
            Assert.AreEqual(Vector3Int.RoundToInt(new Vector3(7f, 0, 2f) + position), Vector3Int.RoundToInt(boxOverlap.GetCellGlobalPosition(14)));
            Assert.AreEqual(Vector3Int.RoundToInt(new Vector3(-7f, 0, -2f) + position), Vector3Int.RoundToInt(boxOverlap.GetCellGlobalPosition(60)));
            Assert.AreEqual(Vector3Int.RoundToInt(new Vector3(7f, 0, -2f) + position), Vector3Int.RoundToInt(boxOverlap.GetCellGlobalPosition(74)));

            Object.DestroyImmediate(testGo);
        }

        [Test]
        public void TestBufferResize()
        {
            List<GameObject> testObjects = new List<GameObject>();
            var testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            testObjects.Add(testGo);
            var boxOverlap = TestBoxOverlapChecker.CreateChecker(agentGameObject: testGo, centerObject: testGo, initialColliderBufferSize: 2, maxColliderBufferSize: 5);
            boxOverlap.Perceive();
            Assert.AreEqual(2, boxOverlap.ColliderBuffer.Length);

            for (var i = 0; i < 3; i++)
            {
                var boxGo = new GameObject("test");
                boxGo.transform.position = Vector3.zero;
                boxGo.AddComponent<BoxCollider>();
                testObjects.Add(boxGo);
            }
            boxOverlap.Perceive();
            Assert.AreEqual(4, boxOverlap.ColliderBuffer.Length);

            for (var i = 0; i < 2; i++)
            {
                var boxGo = new GameObject("test");
                boxGo.transform.position = Vector3.zero;
                boxGo.AddComponent<BoxCollider>();
                testObjects.Add(boxGo);
            }
            boxOverlap.Perceive();
            Assert.AreEqual(5, boxOverlap.ColliderBuffer.Length);

            Object.DestroyImmediate(testGo);
            foreach (var go in testObjects)
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void TestParseCollidersClosest()
        {
            var tag1 = "Player";
            List<GameObject> testObjects = new List<GameObject>();
            var testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            var boxOverlap = TestBoxOverlapChecker.CreateChecker(
                cellScaleX: 10f,
                cellScaleZ: 10f,
                gridSizeX: 2,
                gridSizeZ: 2,
                agentGameObject: testGo,
                centerObject: testGo,
                detectableTags: new[] { tag1 });
            var helper = new VerifyParseCollidersHelper();
            boxOverlap.GridOverlapDetectedClosest += helper.DetectedAction;

            for (var i = 0; i < 3; i++)
            {
                var boxGo = new GameObject("test");
                boxGo.transform.position = new Vector3(i + 1, 0, 1);
                boxGo.AddComponent<BoxCollider>();
                boxGo.tag = tag1;
                testObjects.Add(boxGo);
            }

            boxOverlap.Perceive();
            helper.Verify(1, new List<GameObject> { testObjects[0] });

            Object.DestroyImmediate(testGo);
            foreach (var go in testObjects)
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void TestParseCollidersAll()
        {
            var tag1 = "Player";
            List<GameObject> testObjects = new List<GameObject>();
            var testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            var boxOverlap = TestBoxOverlapChecker.CreateChecker(
                cellScaleX: 10f,
                cellScaleZ: 10f,
                gridSizeX: 2,
                gridSizeZ: 2,
                agentGameObject: testGo,
                centerObject: testGo,
                detectableTags: new[] { tag1 });
            var helper = new VerifyParseCollidersHelper();
            boxOverlap.GridOverlapDetectedAll += helper.DetectedAction;

            for (var i = 0; i < 3; i++)
            {
                var boxGo = new GameObject("test");
                boxGo.transform.position = new Vector3(i + 1, 0, 1);
                boxGo.AddComponent<BoxCollider>();
                boxGo.tag = tag1;
                testObjects.Add(boxGo);
            }

            boxOverlap.Perceive();
            helper.Verify(3, testObjects);

            Object.DestroyImmediate(testGo);
            foreach (var go in testObjects)
            {
                Object.DestroyImmediate(go);
            }
        }

        public class VerifyParseCollidersHelper
        {
            int m_NumInvoked;
            List<GameObject> m_ParsedObjects = new List<GameObject>();

            public void DetectedAction(GameObject go, int cellIndex)
            {
                m_NumInvoked += 1;
                m_ParsedObjects.Add(go);
            }

            public void Verify(int expectNumInvoke, List<GameObject> expectedObjects)
            {
                Assert.AreEqual(expectNumInvoke, m_NumInvoked);
                Assert.AreEqual(expectedObjects.Count, m_ParsedObjects.Count);
                foreach (var obj in expectedObjects)
                {
                    Assert.Contains(obj, m_ParsedObjects);
                }
            }
        }

        [Test]
        public void TestOnlyOneChecker()
        {
            var testGo = new GameObject("test");
            testGo.transform.position = Vector3.zero;
            var gridSensorComponent = testGo.AddComponent<SimpleTestGridSensorComponent>();
            gridSensorComponent.SetComponentParameters(useGridSensorBase: true, useTestingGridSensor: true);
            var sensors = gridSensorComponent.CreateSensors();
            int numChecker = 0;
            foreach (var sensor in sensors)
            {
                var gridsensor = (GridSensorBase)sensor;
                if (gridsensor.m_GridPerception != null)
                {
                    numChecker += 1;
                }
            }
            Assert.AreEqual(1, numChecker);
        }
    }
}
#endif
