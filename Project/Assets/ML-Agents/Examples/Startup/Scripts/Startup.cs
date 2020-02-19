using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace MLAgentsExamples
{
    internal class Startup : MonoBehaviour
    {
        const string k_SceneVariableName = "SCENE_NAME";
        private const string k_SceneCommandLineFlag = "--mlagents-scene-name";

        void Awake()
        {
            var sceneName = "";
            
            // Check for the CLI '--scene-name' flag.  This will be used if
            // no scene environment variable is found.
            var args = Environment.GetCommandLineArgs();
            Console.WriteLine("Command line arguments passed: " + String.Join(" ", args));
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
                    $"You didn't specify the {k_SceneVariableName} environment variable or the {k_SceneCommandLineFlag} command line argument."
                );
                Application.Quit(22);
                return;
            }
            if (SceneUtility.GetBuildIndexByScenePath(sceneName) < 0)
            {
                Console.WriteLine(
                    $"The scene {sceneName} doesn't exist within your build."
                );
                Application.Quit(22);
                return;
            }
            SceneManager.LoadSceneAsync(sceneName);
        }
    }
}
