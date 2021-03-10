using System;
using System.Linq;
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
        private string[] m_AvailableInputSettingsAssets;
        private GUIContent[] m_AvailableSettingsAssetsOptions;
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

        public override void OnTitleBarGUI()
        {
            if (EditorGUILayout.DropdownButton(EditorGUIUtility.IconContent("_Popup"), FocusType.Passive, EditorStyles.label))
            {
                var menu = new GenericMenu();
                menu.AddDisabledItem(new GUIContent("Available Settings Assets:"));
                menu.AddSeparator("");
                for (var i = 0; i < m_AvailableSettingsAssetsOptions.Length; i++)
                    menu.AddItem(new GUIContent(m_AvailableSettingsAssetsOptions[i]), m_CurrentSelectedInputSettingsAsset == i, (path) =>
                    {
                        InputSystem.settings = AssetDatabase.LoadAssetAtPath<InputSettings>((string)path);
                    }, m_AvailableInputSettingsAssets[i]);
                menu.AddSeparator("");
                menu.AddItem(new GUIContent("New Settings Asset…"), false, CreateNewSettingsAsset);
                menu.ShowAsContext();
                Event.current.Use();
            }
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
            // Find the set of available assets in the project.
            m_AvailableInputSettingsAssets = FindSettingsInProject();

            m_Settings = MLAgentsSettings.Instance;
            var currentSettingsPath = AssetDatabase.GetAssetPath(m_Settings);

            if (string.IsNullOrEmpty(currentSettingsPath))
            {
                if (m_AvailableInputSettingsAssets.Length != 0)
                {
                    m_CurrentSelectedInputSettingsAsset = 0;
                    m_Settings = AssetDatabase.LoadAssetAtPath<InputSettings>(m_AvailableInputSettingsAssets[0]);
                    InputSystem.settings = m_Settings;
                }
            }
            else
            {
                m_CurrentSelectedInputSettingsAsset = ArrayHelpers.IndexOf(m_AvailableInputSettingsAssets, currentSettingsPath);
                if (m_CurrentSelectedInputSettingsAsset == -1)
                {
                    // This is odd and shouldn't happen. Solve by just adding the path to the list.
                    m_CurrentSelectedInputSettingsAsset =
                        ArrayHelpers.Append(ref m_AvailableInputSettingsAssets, currentSettingsPath);
                }

                ////REVIEW: should we store this by platform?
                EditorBuildSettings.AddConfigObject(kEditorBuildSettingsConfigKey, m_Settings, true);
            }

            // if (IsSettingsAvailable())
            // {
            //     m_Settings = AssetDatabase.LoadAssetAtPath<MLAgentsSettings>(k_CustomSettingsPath);
            //     MLAgentsSettings.Instance = m_Settings;
            // }
            m_SettingsObject = new SerializedObject(m_Settings);
        }

        private static string[] FindSettingsInProject()
        {
            var guids = AssetDatabase.FindAssets("t:MLAgentsSettings");
            return guids.Select(guid => AssetDatabase.GUIDToAssetPath(guid)).ToArray();
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
