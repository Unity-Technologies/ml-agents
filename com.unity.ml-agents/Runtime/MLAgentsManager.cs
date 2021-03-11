using System;
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
        internal static MLAgentsSettings s_Settings;
        internal static event Action OnSettingsChange;
        internal const string EditorBuildSettingsConfigKey = "com.unity.ml-agents.settings";

        static MLAgentsManager()
        {
#if UNITY_EDITOR
            InitializeInEditor();
#else
            InitializeInPlayer();
#endif
        }

        public static MLAgentsSettings Settings
        {
            get => s_Settings;
            set
            {
                s_Settings = value;
                ApplySettings();
            }
        }

#if UNITY_EDITOR
        // internal static MLAgentsManagerObject s_ManagerObject;
        internal static void InitializeInEditor()
        {
            Debug.Log("InitializeInEditor");

            // See if we have a remembered settings object.
            if (EditorBuildSettings.TryGetConfigObject(EditorBuildSettingsConfigKey,
                out MLAgentsSettings settingsAsset))
            {
                Debug.Log("load from EditorBuildSettings");
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
