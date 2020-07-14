using NUnit.Framework;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if UNITY_2019_3_OR_NEWER
using System.Reflection;
using Unity.MLAgents;
#endif

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class AcademyTests
    {
        [Test]
        public void TestPackageVersion()
        {
            // Make sure that the version strings in the package and Academy don't get out of sync.
            // Unfortunately, the PackageInfo methods don't exist in earlier versions of the editor.
#if UNITY_2019_3_OR_NEWER
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(Agent).Assembly);
            Assert.AreEqual("com.unity.ml-agents", packageInfo.name);
            Assert.AreEqual(Academy.k_PackageVersion, packageInfo.version);
#endif
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
