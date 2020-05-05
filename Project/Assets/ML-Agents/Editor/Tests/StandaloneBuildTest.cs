using System;
using UnityEditor;
using UnityEngine;
using UnityEditor.Build.Reporting;

namespace Unity.MLAgents
{
    public class StandaloneBuildTest
    {
        const string k_OutputCommandLineFlag = "--mlagents-build-output-path";
        const string k_SceneCommandLineFlag = "--mlagents-build-scene-path";

        public static void BuildStandalonePlayerOSX()
        {
            // Read commandline arguments for options
            var outputPath = "testPlayer";
            var scenePath = "Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity";

            var args = Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == k_OutputCommandLineFlag)
                {
                    outputPath = args[i + 1];
                    Debug.Log($"Overriding output path to {outputPath}");
                }
                else if (args[i] == k_SceneCommandLineFlag)
                {
                    scenePath = args[i + 1];
                }
            }

            string[] scenes = { scenePath };
            var buildResult = BuildPipeline.BuildPlayer(
                scenes,
                outputPath,
                BuildTarget.StandaloneOSX,
                BuildOptions.None
            );
            var isOk = buildResult.summary.result == BuildResult.Succeeded;
            var error = "";
            foreach (var stepInfo in buildResult.steps)
            {
                foreach (var msg in stepInfo.messages)
                {
                    if (msg.type != LogType.Log && msg.type != LogType.Warning)
                    {
                        error += msg.content + "\n";
                    }
                }
            }
            if (isOk)
            {
                EditorApplication.Exit(0);
            }
            else
            {
                Console.Error.WriteLine(error);
                EditorApplication.Exit(1);

            }
            Debug.Log(error);

        }

    }
}
