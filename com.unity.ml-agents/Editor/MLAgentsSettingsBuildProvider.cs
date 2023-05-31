using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;


namespace Unity.MLAgents.Editor
{
    internal class MLAgentsSettingsBuildProvider : IPreprocessBuildWithReport, IPostprocessBuildWithReport
    {
        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report)
        {
            if (!EditorUtility.IsPersistent(MLAgentsSettingsManager.Settings))
                return;

            var preloadedAssets = PlayerSettings.GetPreloadedAssets().ToList();
            if (!preloadedAssets.Contains(MLAgentsSettingsManager.Settings))
            {
                preloadedAssets.Add(MLAgentsSettingsManager.Settings);
                PlayerSettings.SetPreloadedAssets(preloadedAssets.ToArray());
            }
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            var preloadedAssets = PlayerSettings.GetPreloadedAssets().ToList();
            if (preloadedAssets.Contains(MLAgentsSettingsManager.Settings))
            {
                preloadedAssets.Remove(MLAgentsSettingsManager.Settings);
                PlayerSettings.SetPreloadedAssets(preloadedAssets.ToArray());
            }
        }
    }
}
