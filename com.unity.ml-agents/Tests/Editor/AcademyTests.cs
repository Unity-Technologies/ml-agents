using NUnit.Framework;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class AcademyTests
    {
        [Test]
        public void TestPackageVersion()
        {
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(Agent).Assembly);
            Assert.AreEqual("com.unity.ml-agents", packageInfo.name);
            Assert.AreEqual(Academy.k_PackageVersion, packageInfo.version);
        }

        class RecursiveAgent : Agent
        {
            int m_collectObsCount;
            public override void CollectObservations(VectorSensor sensor)
            {
                m_collectObsCount++;
                if (m_collectObsCount == 1)
                {
                    // NEVER DO THIS IN REAL CODE!
                    Academy.Instance.EnvironmentStep();
                }
            }
        }

        [Test]
        public void TestRecursiveStepThrows()
        {
            var gameObj = new GameObject();
            var agent = gameObj.AddComponent<RecursiveAgent>();
            agent.LazyInitialize();
            agent.RequestDecision();

            Assert.Throws<UnityAgentsException>(() =>
            {
                Academy.Instance.EnvironmentStep();
            });

            // Make sure the Academy reset to a good state and is still steppable.
            Academy.Instance.EnvironmentStep();
        }


    }
}
