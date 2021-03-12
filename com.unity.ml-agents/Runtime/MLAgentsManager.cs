using System;
using System.Linq;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents
{
#if UNITY_EDITOR
    [InitializeOnLoad]
#endif
    internal static class MLAgentsManager
    {
        internal static event Action OnSettingsChange;
        internal const string EditorBuildSettingsConfigKey = "com.unity.ml-agents.settings";
        private static MLAgentsSettings s_Settings;


        public static MLAgentsSettings Settings
        {
            get => s_Settings;
            set
            {
                EditorBuildSettings.AddConfigObject(EditorBuildSettingsConfigKey, value, true);
                s_Settings = value;
                ApplySettings();
            }
        }

        static MLAgentsManager()
        {
#if UNITY_EDITOR
            InitializeInEditor();
#else
            InitializeInPlayer();
#endif
        }

#if UNITY_EDITOR
        internal static void InitializeInEditor()
        {
            if (EditorBuildSettings.TryGetConfigObject(EditorBuildSettingsConfigKey,
                out MLAgentsSettings settingsAsset))
            {
                Settings = settingsAsset;
            }
        }
#else
        internal static void InitializeInPlayer()
        {
            s_Settings = Resources.FindObjectsOfTypeAll<MLAgentsSettings>().FirstOrDefault() ?? ScriptableObject.CreateInstance<MLAgentsSettings>();
        }
#endif

        internal static void ApplySettings()
        {
            OnSettingsChange?.Invoke();
        }
    }
}
