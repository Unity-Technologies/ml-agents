using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEditor;
using UnityEngine;

namespace Unity.MLAgents
{
    public class SampleExporter
    {
        const string k_MLAgentsSampleFile = "mlagents-sample.json";
        const string k_PackageSampleFile = ".sample.json";
        const string k_MLAgentsDir = "ML-Agents";
        const string k_MLAgentsExamplesDir = "Examples";
        const string k_MLAgentsPackageName = "com.unity.ml-agents";
        const string k_MLAgentsSamplesDirName = "Samples";
        const string k_MLAgentsScriptsDirName = "Scripts";

        struct MLAgentsSampleJson
        {
#pragma warning disable 649
            public string displayName;
            public string description;
            // ReSharper disable once CollectionNeverUpdated.Local
            public List<string> scenes;
#pragma warning restore 649
        }

        struct PackageSampleJson
        {
            public string displayName;
            public string description;
        }

        public static void ExportCuratedSamples()
        {
            var oldBurst = EditorPrefs.GetBool("BurstCompilation");
            EditorPrefs.SetBool("BurstCompilation", false);
            try
            {
                // Path to Project/Assets
                var assetsDir = Application.dataPath;
                var repoRoot = Directory.GetParent(Directory.GetParent(assetsDir).FullName).FullName;

                // Top level of where to store the samples
                var samplesDir = Path.Combine(
                    repoRoot,
                    k_MLAgentsPackageName,
                    k_MLAgentsSamplesDirName);

                if (!Directory.Exists(samplesDir))
                {
                    Directory.CreateDirectory(samplesDir);
                }

                // Path to the examples dir in the project
                var examplesDir = Path.Combine(Application.dataPath, k_MLAgentsDir, k_MLAgentsExamplesDir);
                foreach (var exampleDirectory in Directory.GetDirectories(examplesDir))
                {
                    var mlAgentsSamplePath = Path.Combine(exampleDirectory, k_MLAgentsSampleFile);
                    if (File.Exists(mlAgentsSamplePath))
                    {
                        var sampleJson = JsonConvert.DeserializeObject<MLAgentsSampleJson>(File.ReadAllText(mlAgentsSamplePath));
                        Debug.Log(JsonConvert.SerializeObject(sampleJson));
                        foreach (var scene in sampleJson.scenes)
                        {
                            var scenePath = Path.Combine(exampleDirectory, scene);
                            if (File.Exists(scenePath))
                            {
                                // Create a Sample Directory
                                var currentSampleDir = Directory.CreateDirectory(Path.Combine(samplesDir,
                                    Path.GetFileNameWithoutExtension(scenePath)));


                                var scriptsPath = Path.Combine(exampleDirectory, k_MLAgentsScriptsDirName);
                                Debug.Log($"Scene Path: {scenePath}");
                                var assets = new List<string> { scenePath.Substring(scenePath.IndexOf("Assets")) };
                                if (!Directory.Exists(Path.Combine(scriptsPath)))
                                {
                                    scriptsPath = exampleDirectory;
                                }

                                scriptsPath = scriptsPath.Substring(scriptsPath.IndexOf("Assets"));
                                foreach (var guid in AssetDatabase.FindAssets("t:Script", new[] { scriptsPath }))
                                {
                                    var path = AssetDatabase.GUIDToAssetPath(guid);
                                    assets.Add(path);
                                    Debug.Log($"Adding Asset: {path}");
                                }

                                var packageFilePath = Path.GetFileNameWithoutExtension(scenePath) + ".unitypackage";
                                AssetDatabase.ExportPackage(assets.ToArray(),
                                    Path.Combine(Application.dataPath, packageFilePath),
                                    ExportPackageOptions.IncludeDependencies | ExportPackageOptions.Recurse);

                                // Move the .unitypackage into the samples folder.
                                var packageFileFullPath = Path.Combine(Application.dataPath, packageFilePath);

                                var packageInSamplePath = Path.Combine(currentSampleDir.FullName, packageFilePath);
                                Debug.Log($"Moving {packageFileFullPath} to {packageInSamplePath}");
                                File.Move(packageFileFullPath, packageInSamplePath);

                                // write the .sample.json file to the sample directory
                                File.WriteAllText(Path.Combine(currentSampleDir.FullName, k_PackageSampleFile),
                                    JsonConvert.SerializeObject(new PackageSampleJson
                                    {
                                        description = sampleJson.description,
                                        displayName = sampleJson.displayName
                                    }));
                            }
                        }
                    }
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
