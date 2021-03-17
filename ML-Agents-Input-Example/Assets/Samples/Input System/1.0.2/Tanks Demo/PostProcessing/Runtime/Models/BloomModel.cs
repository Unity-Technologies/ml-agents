using System;

namespace UnityEngine.PostProcessing
{
    [Serializable]
    public class BloomModel : PostProcessingModel
    {
        [Serializable]
        public struct BloomSettings
        {
            [Min(0f), Tooltip("Blend factor of the result image.")]
            public float intensity;

            [Min(0f), Tooltip("Filters out pixels under this level of brightness.")]
            public float threshold;

            public float thresholdLinear
            {
                set { threshold = Mathf.LinearToGammaSpace(value); }
                get { return Mathf.GammaToLinearSpace(threshold); }
            }

            [Range(0f, 1f), Tooltip("Makes transition between under/over-threshold gradual (0 = hard threshold, 1 = soft threshold).")]
            public float softKnee;

            [Range(1f, 7f), Tooltip("Changes extent of veiling effects in a screen resolution-independent fashion.")]
            public float radius;

            [Tooltip("Reduces flashing noise with an additional filter.")]
            public bool antiFlicker;

            public static BloomSettings defaultSettings
            {
                get
                {
                    return new BloomSettings
                    {
                        intensity = 0.5f,
                        threshold = 1.1f,
                        softKnee = 0.5f,
                        radius = 4f,
                        antiFlicker = false,
                    };
                }
            }
        }

        [Serializable]
        public struct LensDirtSettings
        {
            [Tooltip("Dirtiness texture to add smudges or dust to the lens.")]
            public Texture texture;

            [Min(0f), Tooltip("Amount of lens dirtiness.")]
            public float intensity;

            public static LensDirtSettings defaultSettings
            {
                get
                {
                    return new LensDirtSettings
                    {
                        texture = null,
                        intensity = 3f
                    };
                }
            }
        }

        [Serializable]
        public struct Settings
        {
            public BloomSettings bloom;
            public LensDirtSettings lensDirt;

            public static Settings defaultSettings
            {
                get
                {
                    return new Settings
                    {
                        bloom = BloomSettings.defaultSettings,
                        lensDirt = LensDirtSettings.defaultSettings
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
