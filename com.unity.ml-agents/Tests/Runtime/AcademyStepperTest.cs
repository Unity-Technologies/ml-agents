using System.Collections;
using NUnit.Framework;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
using UnityEngine.TestTools;
#endif
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Unity.MLAgents.Tests
{
    public class AcademyStepperTest
    {

        [SetUp]
        public void Setup()
        {
            SceneManager.LoadScene("Packages/com.unity.ml-agents/Tests/Runtime/AcademyStepperTestScene");
        }

        /// <summary>
        /// Test that the serialized agent in the scene, which has its agent parameter value serialized,
        /// properly deserializes it to Agent.maxStep.
        /// </summary>
        [UnityTest]
        public IEnumerator AcademyStepperCleanupPasses()
        {
            int num = 0;
            Transform[] ts0 = (Transform[])Resources.FindObjectsOfTypeAll(typeof(Transform));
            foreach (var t in ts0)
            {
                if (t.GetComponent<AcademyFixedUpdateStepper>() != null)
                {
                    Debug.Log(t.GetComponent<AcademyFixedUpdateStepper>().gameObject.name);
                    num++;
                }
            }
            Debug.Log(num);

            if (SceneManager.GetActiveScene().name != "AcademyStepperTestScene")
            {
                yield return null;
            }

            // Count the number of hidden Fixed Update Steppers
            Transform[] ts = (Transform[])Resources.FindObjectsOfTypeAll(typeof(Transform));
            num = 0;
            foreach (var t in ts)
            {
                if (t.GetComponent<AcademyFixedUpdateStepper>() != null)
                {
                    num++;
                }
            }
            Debug.Log(num);
            Assert.True(num == 1);


        }
    }
}
