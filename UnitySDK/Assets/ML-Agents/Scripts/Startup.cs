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
            SwitchScene(sceneName);
        }
        
        private static void SwitchScene(string sceneName)
        {
            if (sceneName == null)
            {
                throw new ArgumentException("You didn't specified the SCENE_NAME environment variable");
            }
            if (SceneUtility.GetBuildIndexByScenePath(sceneName) < 0)
            {
                throw new ArgumentException("The scene " + sceneName + " doesn't exist within your build. ");
            }
            SceneManager.LoadSceneAsync(sceneName);
        }
    }
}