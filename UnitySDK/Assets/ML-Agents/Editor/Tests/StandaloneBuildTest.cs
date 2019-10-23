using System;
using UnityEditor;
using UnityEngine;

namespace MLAgents
{
    public class StandaloneBuildTest
    {
        static void BuildStandalonePlayerOSX()
        {
            string[] scenes = { "Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity" };
            var error = BuildPipeline.BuildPlayer(scenes, "testPlayer", BuildTarget.StandaloneOSX, BuildOptions.None);
            if (string.IsNullOrEmpty(error))
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
