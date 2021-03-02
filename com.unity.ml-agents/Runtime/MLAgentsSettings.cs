using System;
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


        public int EditorPort
        {
            get { return m_EditorPort; }
            set
            {
                m_EditorPort = value;
            }
        }

        public static MLAgentsSettings Instance
        {
            get
            {
                if (s_Instance == null)
                {
                    var settings = AssetDatabase.LoadAssetAtPath<MLAgentsSettings>(k_CustomSettingsPath);
                    if (settings == null)
                    {
                        settings = ScriptableObject.CreateInstance<MLAgentsSettings>();
                    }
                    s_Instance = settings;
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
