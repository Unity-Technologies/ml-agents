using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.iOS;

namespace Unity.MLAgents
{
    public class SampleExporter
    {
        const string k_OutputCommandLineFlag = "--mlagents-sample-ouput-path";
        const string k_SceneFlag = "--mlagents-scene-path";

        public static void ExportCuratedSamples()
        {
            var oldBurst = EditorPrefs.GetBool("BurstCompilation");
            EditorPrefs.SetBool("BurstCompilation", false);
            try
            {
                var args = Environment.GetCommandLineArgs();
                var outputPath = "exported_samples";
                var scenes = new List<string>();
                for (var i = 0; i < args.Length - 1; i++)
                {
                    if (args[i] == k_OutputCommandLineFlag)
                    {
                        outputPath = args[i + 1];
                        Debug.Log($"Overriding output path to {outputPath}");
                    }
                    if (args[i] == k_SceneFlag)
                    {
                        scenes.Add(args[i + 1]);
                        Debug.Log($"Exporting Scene {scenes.Last()}");
                    }
                }

                foreach (var scene in scenes)
                {
                    var assets = new List<string> { scene };
                    var exampleFolderToAdd = Directory.GetParent(Directory.GetParent(scene).FullName).FullName;
                    Debug.Log($"Parent of Scene: {exampleFolderToAdd}");
                    if (Directory.Exists(Path.Combine(exampleFolderToAdd, "Scripts")))
                    {
                        exampleFolderToAdd = Path.Combine(exampleFolderToAdd, "Scripts");
                    }

                    exampleFolderToAdd = exampleFolderToAdd.Substring(exampleFolderToAdd.IndexOf("Assets"));
                    foreach (var guid in AssetDatabase.FindAssets("t:Script", new[] { exampleFolderToAdd }))
                    {
                        var path = AssetDatabase.GUIDToAssetPath(guid);
                        assets.Add(path);
                        Debug.Log($"Adding Asset: {path}");
                    }
                    AssetDatabase.ExportPackage(assets.ToArray(), Path.GetFileNameWithoutExtension(scene) + ".unitypackage", ExportPackageOptions.IncludeDependencies | ExportPackageOptions.Recurse);
                }
            }
            catch (Exception e)
            {
                Debug.Log(e);
                EditorApplication.Exit(1);
            }
            EditorPrefs.SetBool("BurstCompilation", oldBurst);
            EditorApplication.Exit(0);
        }
    }
}
