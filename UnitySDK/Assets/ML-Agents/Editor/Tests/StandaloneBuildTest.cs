using System;
using UnityEditor;
using UnityEngine;
using UnityEditor.Build.Reporting;

namespace MLAgents
{
    public class StandaloneBuildTest
    {
        static void BuildStandalonePlayerOSX()
        {
            string[] scenes = { "Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity" };
            var report = BuildPipeline.BuildPlayer(scenes, "testPlayer", BuildTarget.StandaloneOSX, BuildOptions.None);
            //  if (string.IsNullOrEmpty(error))
            if(report.summary.result == BuildResult.Succeeded)
            {
                EditorApplication.Exit(0);
            }
            else
            {
                //Console.Error.WriteLine(error);
                EditorApplication.Exit(1);

            }
            //Debug.Log(error);
        }

    }
}
