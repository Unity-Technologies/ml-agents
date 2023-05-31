using System;
using System.Linq;
using System.IO;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEditor;
using UnityEngine.UIElements;

[assembly: InternalsVisibleTo("Unity.ML-Agents.DevTests.Editor")]
namespace Unity.MLAgents.Editor
{
    internal class MLAgentsSettingsProvider : SettingsProvider, IDisposable
    {
        const string k_SettingsPath = "Project/ML-Agents";
        private static MLAgentsSettingsProvider s_Instance;
        private string[] m_AvailableSettingsAssets;
        private int m_CurrentSelectedSettingsAsset;
        private SerializedObject m_SettingsObject;
        [SerializeField]
        private MLAgentsSettings m_Settings;


        private MLAgentsSettingsProvider(string path, SettingsScope scope = SettingsScope.Project)
            : base(path, scope)
        {
            s_Instance = this;
        }

        [SettingsProvider]
        public static SettingsProvider CreateMLAgentsSettingsProvider()
        {
            return new MLAgentsSettingsProvider(k_SettingsPath, SettingsScope.Project);
        }

        public override void OnActivate(string searchContext, VisualElement rootElement)
        {
            base.OnActivate(searchContext, rootElement);
            MLAgentsSettingsManager.OnSettingsChange += Reinitialize;
        }

        public override void OnDeactivate()
        {
            base.OnDeactivate();
            MLAgentsSettingsManager.OnSettingsChange -= Reinitialize;
        }

        public void Dispose()
        {
            m_SettingsObject?.Dispose();
        }

        public override void OnTitleBarGUI()
        {
            if (EditorGUILayout.DropdownButton(EditorGUIUtility.IconContent("_Popup"), FocusType.Passive, EditorStyles.label))
            {
                var menu = new GenericMenu();
                for (var i = 0; i < m_AvailableSettingsAssets.Length; i++)
                {
                    menu.AddItem(ExtractDisplayName(m_AvailableSettingsAssets[i]), m_CurrentSelectedSettingsAsset == i, (path) =>
                    {
                        MLAgentsSettingsManager.Settings = AssetDatabase.LoadAssetAtPath<MLAgentsSettings>((string)path);
                    }, m_AvailableSettingsAssets[i]);
                }
                menu.AddSeparator("");
                menu.AddItem(new GUIContent("New Settings Asset…"), false, CreateNewSettingsAsset);
                menu.ShowAsContext();
                Event.current.Use();
            }
        }

        private GUIContent ExtractDisplayName(string name)
        {
            if (name.StartsWith("Assets/"))
                name = name.Substring("Assets/".Length);
            if (name.EndsWith(".asset"))
                name = name.Substring(0, name.Length - ".asset".Length);
            if (name.EndsWith(".mlagents.settings"))
                name = name.Substring(0, name.Length - ".mlagents.settings".Length);

            // Ugly hack: GenericMenu interprets "/" as a submenu path. But luckily, "/" is not the only slash we have in Unicode.
            return new GUIContent(name.Replace("/", "\u29f8"));
        }

        private void CreateNewSettingsAsset()
        {
            // Asset database always use forward slashes. Use forward slashes for all the paths.
            var projectName = PlayerSettings.productName;
            var path = EditorUtility.SaveFilePanel("Create ML-Agents Settings File", "Assets",
                projectName + ".mlagents.settings", "asset");
            if (string.IsNullOrEmpty(path))
            {
                return;
            }

            path = path.Replace("\\", "/"); // Make sure we only get '/' separators.
            var assetPath = Application.dataPath + "/";
            if (!path.StartsWith(assetPath, StringComparison.CurrentCultureIgnoreCase))
            {
                Debug.LogError(string.Format(
                    "Settings must be stored in Assets folder of the project (got: '{0}')", path));
                return;
            }

            var extension = Path.GetExtension(path);
            if (string.Compare(extension, ".asset", StringComparison.InvariantCultureIgnoreCase) != 0)
            {
                path += ".asset";
            }
            var relativePath = "Assets/" + path.Substring(assetPath.Length);
            CreateNewSettingsAsset(relativePath);
        }

        private static void CreateNewSettingsAsset(string relativePath)
        {
            var settings = ScriptableObject.CreateInstance<MLAgentsSettings>();
            AssetDatabase.CreateAsset(settings, relativePath);
            EditorGUIUtility.PingObject(settings);
            // Install the settings. This will lead to an MLAgentsManager.OnSettingsChange event
            // which in turn will cause this Provider to reinitialize
            MLAgentsSettingsManager.Settings = settings;
        }

        public override void OnGUI(string searchContext)
        {
            if (m_Settings == null)
            {
                InitializeWithCurrentSettings();
            }

            if (m_AvailableSettingsAssets.Length == 0)
            {
                EditorGUILayout.HelpBox(
                    "Click the button below to create a settings asset you can edit.",
                    MessageType.Info);
                if (GUILayout.Button("Create settings asset", GUILayout.Height(30)))
                    CreateNewSettingsAsset();
                GUILayout.Space(20);
            }

            using (new EditorGUI.DisabledScope(m_AvailableSettingsAssets.Length == 0))
            {
                EditorGUI.BeginChangeCheck();
                EditorGUILayout.LabelField("Trainer Settings", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(m_SettingsObject.FindProperty("m_ConnectTrainer"), new GUIContent("Connect to Trainer"));
                EditorGUILayout.PropertyField(m_SettingsObject.FindProperty("m_EditorPort"), new GUIContent("Editor Training Port"));
                EditorGUI.indentLevel--;
                if (EditorGUI.EndChangeCheck())
                    m_SettingsObject.ApplyModifiedProperties();
            }
        }

        internal void InitializeWithCurrentSettings()
        {
            m_AvailableSettingsAssets = FindSettingsInProject();

            m_Settings = MLAgentsSettingsManager.Settings;
            var currentSettingsPath = AssetDatabase.GetAssetPath(m_Settings);
            if (string.IsNullOrEmpty(currentSettingsPath))
            {
                if (m_AvailableSettingsAssets.Length > 0)
                {
                    m_CurrentSelectedSettingsAsset = 0;
                    m_Settings = AssetDatabase.LoadAssetAtPath<MLAgentsSettings>(m_AvailableSettingsAssets[0]);
                    MLAgentsSettingsManager.Settings = m_Settings;
                }
            }
            else
            {
                var settingsList = m_AvailableSettingsAssets.ToList();
                m_CurrentSelectedSettingsAsset = settingsList.IndexOf(currentSettingsPath);

                EditorBuildSettings.AddConfigObject(MLAgentsSettingsManager.EditorBuildSettingsConfigKey, m_Settings, true);
            }

            m_SettingsObject = new SerializedObject(m_Settings);
        }

        private static string[] FindSettingsInProject()
        {
            var guids = AssetDatabase.FindAssets("t:MLAgentsSettings");
            return guids.Select(guid => AssetDatabase.GUIDToAssetPath(guid)).ToArray();
        }

        private void Reinitialize()
        {
            if (m_Settings != null && MLAgentsSettingsManager.Settings != m_Settings)
            {
                InitializeWithCurrentSettings();
            }
            Repaint();
        }
    }
}
