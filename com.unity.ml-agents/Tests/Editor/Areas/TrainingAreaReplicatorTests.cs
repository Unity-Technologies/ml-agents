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
        private TrainingAreaReplicator m_Replicator;

        [SetUp]
        public void Setup()
        {
            var gameObject = new GameObject();
            var trainingArea = new GameObject();
            trainingArea.name = "MyTrainingArea";
            m_Replicator = gameObject.AddComponent<TrainingAreaReplicator>();
            m_Replicator.numAreas = 10;
            m_Replicator.baseArea = trainingArea;
            m_Replicator.Awake();
            m_Replicator.OnEnable();
        }

        [Test]
        public void TestComputeGridSize()
        {
            Assert.GreaterOrEqual(m_Replicator.GridSize.x * m_Replicator.GridSize.y * m_Replicator.GridSize.z, m_Replicator.numAreas);
            Assert.AreEqual(new int3(2, 2, 6), m_Replicator.GridSize);
        }

        [Test]
        public void TestAddEnvironments()
        {
            var trainingAreas = Resources.FindObjectsOfTypeAll<GameObject>().Where(obj => obj.name == m_Replicator.TrainingAreaName);
            Assert.AreEqual(10, trainingAreas.Count());

        }
    }
}
