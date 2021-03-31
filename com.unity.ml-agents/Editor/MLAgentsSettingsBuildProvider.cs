using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;


namespace Unity.MLAgents.Editor
{
    internal class MLAgentsSettingsBuildProvider : IPreprocessBuildWithReport, IPostprocessBuildWithReport
    {
        private MLAgentsSettings m_SettingsAddedToPreloadedAssets;

        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report)
        {
            var wasDirty = IsPlayerSettingsDirty();
            m_SettingsAddedToPreloadedAssets = null;

            var preloadedAssets = PlayerSettings.GetPreloadedAssets().ToList();
            if (!preloadedAssets.Contains(MLAgentsSettingsManager.Settings))
            {
                m_SettingsAddedToPreloadedAssets = MLAgentsSettingsManager.Settings;
                preloadedAssets.Add(m_SettingsAddedToPreloadedAssets);
                PlayerSettings.SetPreloadedAssets(preloadedAssets.ToArray());
            }

            if (!wasDirty)
                ClearPlayerSettingsDirtyFlag();
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            if (m_SettingsAddedToPreloadedAssets == null)
                return;

            var wasDirty = IsPlayerSettingsDirty();

            var preloadedAssets = PlayerSettings.GetPreloadedAssets().ToList();
            if (preloadedAssets.Contains(m_SettingsAddedToPreloadedAssets))
            {
                preloadedAssets.Remove(m_SettingsAddedToPreloadedAssets);
                PlayerSettings.SetPreloadedAssets(preloadedAssets.ToArray());
            }

            m_SettingsAddedToPreloadedAssets = null;

            if (!wasDirty)
                ClearPlayerSettingsDirtyFlag();
        }


        private static bool IsPlayerSettingsDirty()
        {
            var settings = Resources.FindObjectsOfTypeAll<PlayerSettings>();
            if (settings != null && settings.Length > 0)
                return EditorUtility.IsDirty(settings[0]);
            return false;
        }

        private static void ClearPlayerSettingsDirtyFlag()
        {
            var settings = Resources.FindObjectsOfTypeAll<PlayerSettings>();
            if (settings != null && settings.Length > 0)
                EditorUtility.ClearDirty(settings[0]);
        }
    }
}
