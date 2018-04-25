using System;

namespace UnityEngine.PostProcessing
{
    [Serializable]
    public class FogModel : PostProcessingModel
    {
        [Serializable]
        public struct Settings
        {
            [Tooltip("Should the fog affect the skybox?")]
            public bool excludeSkybox;

            public static Settings defaultSettings
            {
                get
                {
                    return new Settings
                    {
                        excludeSkybox = true
                    };
                }
            }
        }

        [SerializeField]
        Settings m_Settings = Settings.defaultSettings;
        public Settings settings
        {
            get { return m_Settings; }
            set { m_Settings = value; }
        }

        public override void Reset()
        {
            m_Settings = Settings.defaultSettings;
        }
    }
}
