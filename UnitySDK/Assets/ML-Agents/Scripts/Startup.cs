using System;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace MLAgents
{
    public class Startup: MonoBehaviour
    {
        void Awake()
        {   
            string sceneName = Environment.GetEnvironmentVariable("SCENE_NAME");
            string controlMode = Environment.GetEnvironmentVariable("CONTROL_MODE");
            if (sceneName != null)
            {
                if (SceneUtility.GetBuildIndexByScenePath(sceneName) >= 0) {
                    SceneManager.LoadSceneAsync(sceneName);
                }
                else
                {
                    throw new ArgumentException("The scene " + sceneName.ToString() + " doesn't exist within your build. ");
                }
            }
            else
            {
                throw new ArgumentException("You didn't specified the SCENE_NAME environment variable");
            }
            if (controlMode != null && controlMode.ToLower() == "true")
            {
                Debug.Log("CONTROL_MODE=true");
                var aca = FindObjectOfType<Academy>();
                if (aca != null)
                {
                    var learningBrains = aca.broadcastHub.broadcastingBrains.Where(
                        x => x != null && x is LearningBrain);
                    foreach (Brain brain in learningBrains)
                    {
                        if (!aca.broadcastHub.IsControlled(brain))
                        {
                            aca.broadcastHub.SetControlled(brain, true);
                        }
                    }
                }
                else
                {
                    throw new ArgumentException("The current scene doesn't have a Academy in it");
                }
            }
            else
            {
                Debug.Log("CONTROL_MODE=false");
            }
        }
    }
}
