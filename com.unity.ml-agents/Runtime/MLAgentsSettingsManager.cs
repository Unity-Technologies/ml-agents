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
    internal static class MLAgentsSettingsManager
    {
        internal static event Action OnSettingsChange;
        internal const string EditorBuildSettingsConfigKey = "com.unity.ml-agents.settings";
        private static MLAgentsSettings s_Settings;


        public static MLAgentsSettings Settings
        {
            get
            {
                if (s_Settings == null)
                {
                    Initialize();
                }
                return s_Settings;
            }
            set
            {
                Debug.Assert(value != null);
#if UNITY_EDITOR
                if (!string.IsNullOrEmpty(AssetDatabase.GetAssetPath(value)))
                {
                    EditorBuildSettings.AddConfigObject(EditorBuildSettingsConfigKey, value, true);
                }
#endif
                s_Settings = value;
                ApplySettings();
            }
        }

        static MLAgentsSettingsManager()
        {
            Initialize();
        }

        static void Initialize()
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
            var settings = ScriptableObject.CreateInstance<MLAgentsSettings>();
            if (EditorBuildSettings.TryGetConfigObject(EditorBuildSettingsConfigKey,
                out MLAgentsSettings settingsAsset))
            {
                if (settingsAsset != null)
                {
                    settings = settingsAsset;
                }
            }
            Settings = settings;
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
