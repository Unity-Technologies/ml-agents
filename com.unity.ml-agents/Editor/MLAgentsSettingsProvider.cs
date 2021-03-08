using System;
using System.IO;
using UnityEngine;
using UnityEditor;
#if UNITY_2019_4_OR_NEWER
using UnityEngine.UIElements;
#else
using UnityEngine.Experimental.UIElements;
#endif

namespace Unity.MLAgents.Editor
{
    internal class MLAgentsSettingsProvider : SettingsProvider, IDisposable
    {
        const string k_SettingsPath = "Project/ML Agents";
        const string k_CustomSettingsPath = "Assets/MLAgents.settings.asset";
        private static MLAgentsSettingsProvider s_Instance;
        private SerializedObject m_SettingsObject;
        [SerializeField]
        private MLAgentsSettings m_Settings;


        private MLAgentsSettingsProvider(string path, SettingsScope scope = SettingsScope.Project)
            : base(path, scope)
        {
            s_Instance = this;
        }

        public override void OnActivate(string searchContext, VisualElement rootElement)
        {
            base.OnActivate(searchContext, rootElement);
            MLAgentsSettings.OnSettingsChange += OnSettingsChange;
        }

        public override void OnDeactivate()
        {
            base.OnDeactivate();
            MLAgentsSettings.OnSettingsChange -= OnSettingsChange;
        }

        public void Dispose()
        {
            m_SettingsObject?.Dispose();
        }

        [SettingsProvider]
        public static SettingsProvider CreateMLAgentsSettingsProvider()
        {
            return new MLAgentsSettingsProvider(k_SettingsPath, SettingsScope.Project);
        }

        public static bool IsSettingsAvailable()
        {
            return File.Exists(k_CustomSettingsPath);
        }

        public override void OnGUI(string searchContext)
        {
            if (m_Settings == null)
            {
                InitializeWithCurrentSettings();
            }

            if (!IsSettingsAvailable())
            {
                EditorGUILayout.HelpBox(
                    "Click the button below to create a settings asset you can edit.",
                    MessageType.Info);
                if (GUILayout.Button("Create settings asset", GUILayout.Height(30)))
                    CreateNewSettingsAsset(k_CustomSettingsPath);
                GUILayout.Space(20);
            }

            using (new EditorGUI.DisabledScope(!IsSettingsAvailable()))
            {
                EditorGUI.BeginChangeCheck();
                EditorGUILayout.PropertyField(m_SettingsObject.FindProperty("m_ConnectTrainer"), new GUIContent("Connect to Trainer"));
                EditorGUILayout.PropertyField(m_SettingsObject.FindProperty("m_EditorPort"), new GUIContent("Editor Port"));
                if (EditorGUI.EndChangeCheck())
                    m_SettingsObject.ApplyModifiedProperties();
            }
        }

        private static void CreateNewSettingsAsset(string relativePath)
        {
            var settings = ScriptableObject.CreateInstance<MLAgentsSettings>();
            AssetDatabase.CreateAsset(settings, relativePath);
            EditorGUIUtility.PingObject(settings);
            // Install the settings. This will lead to an MLAgentsSettings.OnSettingsChange event which in turn
            // will cause us to re-initialize.
            MLAgentsSettings.Instance = settings;
        }

        internal void InitializeWithCurrentSettings()
        {
            m_Settings = MLAgentsSettings.Instance;
            var currentSettingsPath = AssetDatabase.GetAssetPath(m_Settings);
            if (IsSettingsAvailable())
            {
                m_Settings = AssetDatabase.LoadAssetAtPath<MLAgentsSettings>(k_CustomSettingsPath);
                MLAgentsSettings.Instance = m_Settings;
            }
            m_SettingsObject = new SerializedObject(m_Settings);
        }

        private void OnSettingsChange()
        {
            if (m_Settings != null && MLAgentsSettings.Instance != m_Settings)
            {
                InitializeWithCurrentSettings();
            }
        }
    }
}
