using System;
using System.Linq;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents
{
    class MLAgentsSettings : ScriptableObject
    {
        private static MLAgentsSettings s_Instance;
        internal static event Action OnSettingsChange;

        const string k_CustomSettingsPath = "Assets/MLAgents.settings.asset";

        [SerializeField]
        private int m_EditorPort = 5004;
        [SerializeField]
        private int m_PlayerPort = -1;


        public int EditorPort
        {
            get { return m_EditorPort; }
            set
            {
                m_EditorPort = value;
            }
        }
        public int PlayerPort
        {
            get { return m_PlayerPort; }
            set
            {
                m_PlayerPort = value;
            }
        }

        public static MLAgentsSettings Instance
        {
            get
            {
                if (s_Instance == null)
                {
#if UNITY_EDITOR
                    var settings = AssetDatabase.LoadAssetAtPath<MLAgentsSettings>(k_CustomSettingsPath);
                    if (settings == null)
                    {
                        settings = ScriptableObject.CreateInstance<MLAgentsSettings>();
                    }
                    s_Instance = settings;
#else
                    s_Instance = Resources.FindObjectsOfTypeAll<MLAgentsSettings>().FirstOrDefault() ?? ScriptableObject.CreateInstance<MLAgentsSettings>();
#endif
                }
                return s_Instance;
            }
            set
            {
                s_Instance = value;
                s_Instance.OnChange();
            }
        }

        internal void OnChange()
        {
            if (MLAgentsSettings.Instance == this)
                OnSettingsChange.Invoke();
        }
    }
}
