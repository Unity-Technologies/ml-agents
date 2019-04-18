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
            SwitchScene(sceneName);
            SwitchControlMode(controlMode);
        }

        private void SwitchScene(string sceneName)
        {
            if (sceneName == null)
            {
                throw new ArgumentException("You didn't specified the SCENE_NAME environment variable");
            }
            else
            {
                if (SceneUtility.GetBuildIndexByScenePath(sceneName) >= 0) {
                    SceneManager.LoadSceneAsync(sceneName);
                }
                else
                {
                    throw new ArgumentException("The scene " + sceneName.ToString() + " doesn't exist within your build. ");
                }
            }
        }

        private void SwitchControlMode(string controlMode)
        {
            bool controlModeBoolean = controlMode != null || controlMode.ToLower() == "true";
            Debug.Log("CONTROL_MODE=" + controlMode);
            var aca = FindObjectOfType<Academy>();
            if (aca != null)
            {
                var learningBrains = aca.broadcastHub.broadcastingBrains.Where(
                    x => x != null && x is LearningBrain);
                foreach (Brain brain in learningBrains)
                {
                    aca.broadcastHub.SetControlled(brain, controlModeBoolean);
                }
            }
            else
            {
                throw new ArgumentException("The current scene doesn't have a Academy in it");
            }
        }
    }
}
