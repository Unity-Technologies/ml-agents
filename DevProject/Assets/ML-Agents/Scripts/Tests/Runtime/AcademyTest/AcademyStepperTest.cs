using System.Collections;
using NUnit.Framework;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
using UnityEngine.TestTools;
#endif
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents;

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
        /// Test that in each FixUpdate(), the Academy is only stepped once.
        /// </summary>
        [UnityTest]
        public IEnumerator AcademyStepperCleanupPasses()
        {
            var academy = Academy.Instance;
            int stepCount1 = academy.TotalStepCount;
            yield return null;
            int stepCount2 = academy.TotalStepCount;
            Assert.True(stepCount2 - stepCount1 == 1);
        }
    }
}
