#if UNITY_CLOUD_BUILD

using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using System.IO;

namespace MLAgents
{
    public static class BuilderUtils
    {
        public static void SwitchAllLearningBrainToControlMode()
        {
            Debug.Log("The Switching to control mode function is triggered");
            string[] scenePaths = Directory.GetFiles("Assets/ML-Agents/Examples/", "*.unity", SearchOption.AllDirectories);
            foreach (string scenePath in scenePaths)
            {
                var curScene = EditorSceneManager.OpenScene(scenePath);
                var aca = SceneAsset.FindObjectOfType<Academy>();
                if (aca != null)
                {
                    var learningBrains = aca.broadcastHub.broadcastingBrains.Where(
                        x => x != null && x is LearningBrain);
                    foreach (Brain brain in learningBrains)
                    {
                        if (!aca.broadcastHub.IsControlled(brain))
                        {
                            Debug.Log("Switched brain in scene " + scenePath);
                            aca.broadcastHub.SetControlled(brain, true);
                        }
                    }
                    EditorSceneManager.SaveScene(curScene);
                }
                else
                {
                    Debug.Log("scene " + scenePath + " doesn't have a Academy in it");
                }
            }
        }
    }
}
    
#endif
