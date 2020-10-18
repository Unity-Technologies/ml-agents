using System.Collections;
using NUnit.Framework;
using UnityEngine.TestTools;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

namespace Tests
{
    public class AcademyStepperTest
    {
        [SetUp]
        public void Setup()
        {
            SceneManager.LoadScene("ML-Agents/Scripts/Tests/Runtime/AcademyTest/AcademyStepperTestScene");
        }

        /// <summary>
        /// Verify in each update, the Academy is only stepped once.
        /// </summary>
        [UnityTest]
        public IEnumerator AcademyStepperCleanupPasses()
        {
            var academy = Academy.Instance;
            int initStepCount = academy.TotalStepCount;
            for (var i = 0; i < 5; i++)
            {
                yield return new WaitForFixedUpdate();
                Assert.True(academy.TotalStepCount - initStepCount == i + 1);
            }
        }
    }
}
