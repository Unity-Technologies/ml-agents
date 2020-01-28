using System;
using UnityEditor;
using UnityEngine;
using UnityEditor.Build.Reporting;

namespace MLAgents
{
    public class StandaloneBuildTest
    {
        public static void BuildStandalonePlayerOSX()
        {
            string[] scenes = { "Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity" };
            var buildResult = BuildPipeline.BuildPlayer(scenes, "testPlayer", BuildTarget.StandaloneOSX, BuildOptions.None);
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
