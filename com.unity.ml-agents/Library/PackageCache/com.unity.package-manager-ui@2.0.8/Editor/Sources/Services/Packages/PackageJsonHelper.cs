using System.IO;
using UnityEngine;

namespace UnityEditor.PackageManager.UI
{
    internal class PackageJsonHelper
    {
        [SerializeField]
        private string name = string.Empty;

        private string path = string.Empty;

        public static string GetPackagePath(string jsonPath)
        {
            return Path.GetDirectoryName(jsonPath).Replace("\\", "/");
        }

        public static PackageJsonHelper Load(string path)
        {
            // If the path is a directory, find the `package.json` file path
            var jsonPath = Directory.Exists(path) ? Path.Combine(path, "package.json") : path;
            if (!File.Exists(jsonPath))
                return null;
            var packageJson = JsonUtility.FromJson<PackageJsonHelper>(File.ReadAllText(jsonPath));
            packageJson.path = GetPackagePath(jsonPath);
            return string.IsNullOrEmpty(packageJson.name) ? null : packageJson;
        }

        public PackageInfo PackageInfo
        {
            get { return new PackageInfo {PackageId = string.Format("{0}@file:{1}", name, path)}; }
        }
    }
}