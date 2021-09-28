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
        [Test]
        public void TestComputeGridSize()
        {
            var gameObject = new GameObject();
            var trainingAreaReplicator = gameObject.AddComponent<TrainingAreaReplicator>();
            trainingAreaReplicator.numAreas = 10;
            trainingAreaReplicator.Awake();
            Assert.AreEqual(new int3(3, 3, 1), trainingAreaReplicator.GridSize);
        }

        [Test]
        public void TestAddEnvironments()
        {
            var gameObject = new GameObject();
            var trainingArea = new GameObject();
            trainingArea.name = "MyTrainingArea";
            var trainingAreaReplicator = gameObject.AddComponent<TrainingAreaReplicator>();
            trainingAreaReplicator.numAreas = 10;
            trainingAreaReplicator.baseArea = trainingArea;
            trainingAreaReplicator.Awake();
            trainingAreaReplicator.OnEnable();
            var trainingAreas = Resources.FindObjectsOfTypeAll<GameObject>().Where(obj => obj.name == trainingAreaReplicator.TrainingAreaName);
            Assert.AreEqual(10, trainingAreas.Count());

        }
    }
}
