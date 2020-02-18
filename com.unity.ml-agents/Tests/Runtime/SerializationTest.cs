// using System.Collections;
// using NUnit.Framework;
// #if UNITY_EDITOR
// using UnityEditor.SceneManagement;
// using UnityEngine.TestTools;
// #endif
// using UnityEngine;
// using UnityEngine.SceneManagement;
//
// namespace Tests
// {
//     [TestFixture]
//     public class SerializationTest
//     {
//
//         [SetUp]
//         public void Setup()
//         {
//             SceneManager.LoadScene("Packages/com.unity.ml-agents/Tests/Runtime/SerializeTestScene");
//         }
//
//         /// <summary>
//         /// Test that the serialized agent in the scene, which has its agent parameter value serialized,
//         /// properly deserializes it to Agent.maxStep.
//         /// </summary>
//         [UnityTest]
//         public IEnumerator SerializationTestSimplePasses()
//         {
//             // Use the Assert class to test conditions
//             var gameObjects = SceneManager.GetActiveScene().GetRootGameObjects();
//             if (SceneManager.GetActiveScene().name != "SerializeTestScene")
//             {
//                 yield return null;
//             }
//
//             GameObject agent = null;
//             foreach (var go in gameObjects)
//             {
//                 if (go.name == "Agent")
//                 {
//                     agent = go;
//                     break;
//                 }
//             }
//             Assert.NotNull(agent);
//             var agentComponent = agent.GetComponent<SerializeAgent>();
//             Assert.True(agentComponent.maxStep == 5000);
//
//         }
//     }
// }
