using System;

namespace UnityEngine.PostProcessing
{
    [Serializable]
    public class ChromaticAberrationModel : PostProcessingModel
    {
        [Serializable]
        public struct Settings
        {
            [Tooltip("Shift the hue of chromatic aberrations.")]
            public Texture2D spectralTexture;

            [Range(0f, 1f), Tooltip("Amount of tangential distortion.")]
            public float intensity;

            public static Settings defaultSettings
            {
                get
                {
                    return new Settings
                    {
                        spectralTexture = null,
                        intensity = 0.1f
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
