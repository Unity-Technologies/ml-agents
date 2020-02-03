using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace MLAgents
{
    public class Startup : MonoBehaviour
    {
        const string k_SceneVariableName = "SCENE_NAME";

        void Awake()
        {
            var sceneName = Environment.GetEnvironmentVariable(k_SceneVariableName);
            SwitchScene(sceneName);
        }

        static void SwitchScene(string sceneName)
        {
            if (sceneName == null)
            {
                throw new ArgumentException(
                    $"You didn't specified the {k_SceneVariableName} environment variable");
            }
            if (SceneUtility.GetBuildIndexByScenePath(sceneName) < 0)
            {
                throw new ArgumentException(
                    $"The scene {sceneName} doesn't exist within your build. ");
            }
            SceneManager.LoadSceneAsync(sceneName);
        }
    }
}
