using System;

namespace UnityEngine.PostProcessing
{
    [Serializable]
    public class ColorGradingModel : PostProcessingModel
    {
        public enum Tonemapper
        {
            None,

            /// <summary>
            /// ACES Filmic reference tonemapper.
            /// </summary>
            ACES,

            /// <summary>
            /// Neutral tonemapper (based off John Hable's & Jim Hejl's work).
            /// </summary>
            Neutral
        }

        [Serializable]
        public struct TonemappingSettings
        {
            [Tooltip("Tonemapping algorithm to use at the end of the color grading process. Use \"Neutral\" if you need a customizable tonemapper or \"Filmic\" to give a standard filmic look to your scenes.")]
            public Tonemapper tonemapper;

            // Neutral settings
            [Range(-0.1f, 0.1f)]
            public float neutralBlackIn;

            [Range(1f, 20f)]
            public float neutralWhiteIn;

            [Range(-0.09f, 0.1f)]
            public float neutralBlackOut;

            [Range(1f, 19f)]
            public float neutralWhiteOut;

            [Range(0.1f, 20f)]
            public float neutralWhiteLevel;

            [Range(1f, 10f)]
            public float neutralWhiteClip;

            public static TonemappingSettings defaultSettings
            {
                get
                {
                    return new TonemappingSettings
                    {
                        tonemapper = Tonemapper.Neutral,

                        neutralBlackIn = 0.02f,
                        neutralWhiteIn = 10f,
                        neutralBlackOut = 0f,
                        neutralWhiteOut = 10f,
                        neutralWhiteLevel = 5.3f,
                        neutralWhiteClip = 10f
                    };
                }
            }
        }

        [Serializable]
        public struct BasicSettings
        {
            [Tooltip("Adjusts the overall exposure of the scene in EV units. This is applied after HDR effect and right before tonemapping so it won't affect previous effects in the chain.")]
            public float postExposure;

            [Range(-100f, 100f), Tooltip("Sets the white balance to a custom color temperature.")]
            public float temperature;

            [Range(-100f, 100f), Tooltip("Sets the white balance to compensate for a green or magenta tint.")]
            public float tint;

            [Range(-180f, 180f), Tooltip("Shift the hue of all colors.")]
            public float hueShift;

            [Range(0f, 2f), Tooltip("Pushes the intensity of all colors.")]
            public float saturation;

            [Range(0f, 2f), Tooltip("Expands or shrinks the overall range of tonal values.")]
            public float contrast;

            public static BasicSettings defaultSettings
            {
                get
                {
                    return new BasicSettings
                    {
                        postExposure = 0f,

                        temperature = 0f,
                        tint = 0f,

                        hueShift = 0f,
                        saturation = 1f,
                        contrast = 1f,
                    };
                }
            }
        }

        [Serializable]
        public struct ChannelMixerSettings
        {
            public Vector3 red;
            public Vector3 green;
            public Vector3 blue;

            [HideInInspector]
            public int currentEditingChannel; // Used only in the editor

            public static ChannelMixerSettings defaultSettings
            {
                get
                {
                    return new ChannelMixerSettings
                    {
                        red = new Vector3(1f, 0f, 0f),
                        green = new Vector3(0f, 1f, 0f),
                        blue = new Vector3(0f, 0f, 1f),
                        currentEditingChannel = 0
                    };
                }
            }
        }

        [Serializable]
        public struct LogWheelsSettings
        {
            [Trackball("GetSlopeValue")]
            public Color slope;

            [Trackball("GetPowerValue")]
            public Color power;

            [Trackball("GetOffsetValue")]
            public Color offset;

            public static LogWheelsSettings defaultSettings
            {
                get
                {
                    return new LogWheelsSettings
                    {
                        slope = Color.clear,
                        power = Color.clear,
                        offset = Color.clear
                    };
                }
            }
        }

        [Serializable]
        public struct LinearWheelsSettings
        {
            [Trackball("GetLiftValue")]
            public Color lift;

            [Trackball("GetGammaValue")]
            public Color gamma;

            [Trackball("GetGainValue")]
            public Color gain;

            public static LinearWheelsSettings defaultSettings
            {
                get
                {
                    return new LinearWheelsSettings
                    {
                        lift = Color.clear,
                        gamma = Color.clear,
                        gain = Color.clear
                    };
                }
            }
        }

	    public enum ColorWheelMode
	    {
		    Linear,
			Log
	    }

        [Serializable]
        public struct ColorWheelsSettings
        {
	        public ColorWheelMode mode;

            [TrackballGroup]
            public LogWheelsSettings log;

            [TrackballGroup]
            public LinearWheelsSettings linear;

            public static ColorWheelsSettings defaultSettings
            {
                get
                {
                    return new ColorWheelsSettings
                    {
						mode = ColorWheelMode.Log,
                        log = LogWheelsSettings.defaultSettings,
                        linear = LinearWheelsSettings.defaultSettings
                    };
                }
            }
        }

        [Serializable]
        public struct CurvesSettings
        {
            public ColorGradingCurve master;
            public ColorGradingCurve red;
            public ColorGradingCurve green;
            public ColorGradingCurve blue;
            public ColorGradingCurve hueVShue;
            public ColorGradingCurve hueVSsat;
            public ColorGradingCurve satVSsat;
            public ColorGradingCurve lumVSsat;

            // Used only in the editor
            [HideInInspector] public int e_CurrentEditingCurve;
            [HideInInspector] public bool e_CurveY;
            [HideInInspector] public bool e_CurveR;
            [HideInInspector] public bool e_CurveG;
            [HideInInspector] public bool e_CurveB;

            public static CurvesSettings defaultSettings
            {
                get
                {
                    return new CurvesSettings
                    {
                        master = new ColorGradingCurve(new AnimationCurve(new Keyframe(0f, 0f, 1f, 1f), new Keyframe(1f, 1f, 1f, 1f)), 0f, false, new Vector2(0f, 1f)),
                        red    = new ColorGradingCurve(new AnimationCurve(new Keyframe(0f, 0f, 1f, 1f), new Keyframe(1f, 1f, 1f, 1f)), 0f, false, new Vector2(0f, 1f)),
                        green  = new ColorGradingCurve(new AnimationCurve(new Keyframe(0f, 0f, 1f, 1f), new Keyframe(1f, 1f, 1f, 1f)), 0f, false, new Vector2(0f, 1f)),
                        blue   = new ColorGradingCurve(new AnimationCurve(new Keyframe(0f, 0f, 1f, 1f), new Keyframe(1f, 1f, 1f, 1f)), 0f, false, new Vector2(0f, 1f)),

                        hueVShue = new ColorGradingCurve(new AnimationCurve(), 0.5f, true,  new Vector2(0f, 1f)),
                        hueVSsat = new ColorGradingCurve(new AnimationCurve(), 0.5f, true,  new Vector2(0f, 1f)),
                        satVSsat = new ColorGradingCurve(new AnimationCurve(), 0.5f, false, new Vector2(0f, 1f)),
                        lumVSsat = new ColorGradingCurve(new AnimationCurve(), 0.5f, false, new Vector2(0f, 1f)),

                        e_CurrentEditingCurve = 0,
                        e_CurveY = true,
                        e_CurveR = false,
                        e_CurveG = false,
                        e_CurveB = false
                    };
                }
            }
        }

        [Serializable]
        public struct Settings
        {
            public TonemappingSettings tonemapping;
            public BasicSettings basic;
            public ChannelMixerSettings channelMixer;
            public ColorWheelsSettings colorWheels;
            public CurvesSettings curves;

            public static Settings defaultSettings
            {
                get
                {
                    return new Settings
                    {
                        tonemapping = TonemappingSettings.defaultSettings,
                        basic = BasicSettings.defaultSettings,
                        channelMixer = ChannelMixerSettings.defaultSettings,
                        colorWheels = ColorWheelsSettings.defaultSettings,
                        curves = CurvesSettings.defaultSettings
                    };
                }
            }
        }

        [SerializeField]
        Settings m_Settings = Settings.defaultSettings;
        public Settings settings
        {
            get { return m_Settings; }
            set
            {
                m_Settings = value;
                OnValidate();
            }
        }

        public bool isDirty { get; internal set; }
        public RenderTexture bakedLut { get; internal set; }

        public override void Reset()
        {
            m_Settings = Settings.defaultSettings;
            OnValidate();
        }

        public override void OnValidate()
        {
            isDirty = true;
        }
    }
}
