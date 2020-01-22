using UnityEngine;
using UnityEditor;
using UnityEditor.Callbacks;
using System.IO;


namespace TMPro
{
    public class TMP_PostBuildProcessHandler
    {
        [PostProcessBuildAttribute(10000)]
        public static void OnPostprocessBuild(BuildTarget target, string pathToBuiltProject)
        {
            // Check if TMP Essential Resource are present in user project.
            if (target == BuildTarget.iOS && File.Exists(GetEssentialProjectResourcesPath() + "/Resources/TMP Settings.asset") && TMP_Settings.enableEmojiSupport)
            {
                string file = Path.Combine(pathToBuiltProject, "Classes/UI/Keyboard.mm");
                string content = File.ReadAllText(file);
                content = content.Replace("FILTER_EMOJIS_IOS_KEYBOARD 1", "FILTER_EMOJIS_IOS_KEYBOARD 0");
                File.WriteAllText(file, content);
            }
        }


        private static string GetEssentialProjectResourcesPath()
        {
            // Find the potential location of the TextMesh Pro folder in the user project.
            string projectPath = Path.GetFullPath("Assets/..");
            if (Directory.Exists(projectPath))
            {
                // Search for default location of TMP Essential Resources
                if (Directory.Exists(projectPath + "/Assets/TextMesh Pro/Resources"))
                {
                    return "Assets/TextMesh Pro";
                }

                // Search for potential alternative locations in the user project
                string[] matchingPaths = Directory.GetDirectories(projectPath, "TextMesh Pro", SearchOption.AllDirectories);
                projectPath = ValidateLocation(matchingPaths, projectPath);
                if (projectPath != null) return projectPath;
            }

            return null;
        }


        private static string ValidateLocation(string[] paths, string projectPath)
        {
            for (int i = 0; i < paths.Length; i++)
            {
                // Check if any of the matching directories contain a GUISkins directory.
                if (Directory.Exists(paths[i] + "/Resources"))
                {
                    string folderPath = paths[i].Replace(projectPath, "");
                    folderPath = folderPath.TrimStart('\\', '/');
                    return folderPath;
                }
            }

            return null;
        }
    }
}
