using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace Tests
{
    [TestFixture]
    public class SerializationTest
    {

        [SetUp]
        public void Setup()
        {
            SceneManager.LoadScene("SerializeTestScene");
        }

        /// <summary>
        /// Test that the serialized agent in the scene, which has its agent parameter value serialized,
        /// properly deserializes it to Agent.maxStep.
        /// </summary>
        [UnityTest]
        public IEnumerator SerializationTestSimplePasses()
        {
            // Use the Assert class to test conditions
            var gameObjects = SceneManager.GetActiveScene().GetRootGameObjects();
            if (SceneManager.GetActiveScene().name != "SerializeTestScene")
            {
                yield return null;
            }

            GameObject agent = null;
            foreach (var go in gameObjects)
            {
                if (go.name == "Agent")
                {
                    agent = go;
                    break;
                }
            }
            Assert.NotNull(agent);
            var agentComponent = agent.GetComponent<SerializeAgent>();
            Assert.True(agentComponent.maxStep == 5000);

        }
    }
}
