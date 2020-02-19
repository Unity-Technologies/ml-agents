using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace MLAgentsExamples
{
    internal class Startup : MonoBehaviour
    {
        const string k_SceneVariableName = "SCENE_NAME";
        private const string k_SceneCommandLineFlag = "--scene-name";

        void Awake()
        {
            var sceneName = "";
            
            // Check for the CLI '--scene-name' flag.  This will be used if
            // no scene environment variable is found.
            var args = Environment.GetCommandLineArgs();
            Console.WriteLine(String.Join(" ", args));
            for (int i = 0; i < args.Length; i++) {
                if (args [i] == k_SceneCommandLineFlag && i < args.Length - 1) {
                    sceneName = args[i + 1];
                }
            }

            var sceneEnvironmentVariable = Environment.GetEnvironmentVariable(k_SceneVariableName);
            if (!string.IsNullOrEmpty(sceneEnvironmentVariable))
            {
                sceneName = sceneEnvironmentVariable;
            }
            
            SwitchScene(sceneName);
        }

        static void SwitchScene(string sceneName)
        {
            if (sceneName == null)
            {
                Console.WriteLine(
                    $"You didn't specified the {k_SceneVariableName} environment variable."
                );
                Application.Quit();
                return;
            }
            if (SceneUtility.GetBuildIndexByScenePath(sceneName) < 0)
            {
                Console.WriteLine(
                    $"The scene {sceneName} doesn't exist within your build."
                );
                Application.Quit();
                return;
            }
            SceneManager.LoadSceneAsync(sceneName);
        }
    }
}
