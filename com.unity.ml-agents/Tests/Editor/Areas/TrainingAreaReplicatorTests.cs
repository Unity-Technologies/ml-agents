using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.MLAgents.Areas;
using UnityEngine;

namespace Unity.MLAgents.Tests.Areas
{
    [TestFixture]
    public class TrainingAreaReplicatorTests
    {
        TrainingAreaReplicator m_Replicator;

        [SetUp]
        public void Setup()
        {
            var gameObject = new GameObject();
            var trainingArea = new GameObject();
            trainingArea.name = "MyTrainingArea";
            m_Replicator = gameObject.AddComponent<TrainingAreaReplicator>();
            m_Replicator.baseArea = trainingArea;
        }

        [TearDown]
        public void TearDown()
        {
            var trainingAreas = Resources.FindObjectsOfTypeAll<GameObject>().Where(obj => obj.name == m_Replicator.TrainingAreaName);
            foreach (var trainingArea in trainingAreas)
            {
                Object.DestroyImmediate(trainingArea);
            }
            m_Replicator = null;
        }

        private static object[] NumAreasCases =
        {
            new object[] {1},
            new object[] {2},
            new object[] {5},
            new object[] {7},
            new object[] {8},
            new object[] {64},
            new object[] {63},
        };

        [TestCaseSource(nameof(NumAreasCases))]
        public void TestComputeGridSize(int numAreas)
        {
            m_Replicator.numAreas = numAreas;
            m_Replicator.Awake();
            m_Replicator.OnEnable();
            var m_CorrectGridSize = int3.zero;
            var m_RootNumAreas = Mathf.Pow(numAreas, 1.0f / 3.0f);
            m_CorrectGridSize.x = Mathf.CeilToInt(m_RootNumAreas);
            m_CorrectGridSize.y = Mathf.CeilToInt(m_RootNumAreas);
            m_CorrectGridSize.z = Mathf.CeilToInt((float)numAreas / (m_CorrectGridSize.x * m_CorrectGridSize.y));
            Assert.GreaterOrEqual(m_Replicator.GridSize.x * m_Replicator.GridSize.y * m_Replicator.GridSize.z, m_Replicator.numAreas);
            Assert.AreEqual(m_CorrectGridSize, m_Replicator.GridSize);
        }

        [Test]
        public void TestAddEnvironments()
        {
            m_Replicator.numAreas = 10;
            m_Replicator.buildOnly = false;
            m_Replicator.Awake();
            m_Replicator.OnEnable();
            var trainingAreas = Resources.FindObjectsOfTypeAll<GameObject>().Where(obj => obj.name == m_Replicator.TrainingAreaName);
            Assert.AreEqual(10, trainingAreas.Count());

        }

        [Test]
        public void TestAddEnvironmentsBuildOnly()
        {
            m_Replicator.numAreas = 10;
            m_Replicator.buildOnly = true;
            m_Replicator.Awake();
            m_Replicator.OnEnable();
            var trainingAreas = Resources.FindObjectsOfTypeAll<GameObject>().Where(obj => obj.name == m_Replicator.TrainingAreaName);
            Assert.AreEqual(1, trainingAreas.Count());

        }
    }
}
