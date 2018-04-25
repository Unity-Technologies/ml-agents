using System;

namespace UnityEngine.PostProcessing
{
    [Serializable]
    public class ScreenSpaceReflectionModel : PostProcessingModel
    {
        public enum SSRResolution
        {
            High = 0,
            Low = 2
        }

        public enum SSRReflectionBlendType
        {
            PhysicallyBased,
            Additive
        }

        [Serializable]
        public struct IntensitySettings
        {
            [Tooltip("Nonphysical multiplier for the SSR reflections. 1.0 is physically based.")]
            [Range(0.0f, 2.0f)]
            public float reflectionMultiplier;

            [Tooltip("How far away from the maxDistance to begin fading SSR.")]
            [Range(0.0f, 1000.0f)]
            public float fadeDistance;

            [Tooltip("Amplify Fresnel fade out. Increase if floor reflections look good close to the surface and bad farther 'under' the floor.")]
            [Range(0.0f, 1.0f)]
            public float fresnelFade;

            [Tooltip("Higher values correspond to a faster Fresnel fade as the reflection changes from the grazing angle.")]
            [Range(0.1f, 10.0f)]
            public float fresnelFadePower;
        }

        [Serializable]
        public struct ReflectionSettings
        {
            // When enabled, we just add our reflections on top of the existing ones. This is physically incorrect, but several
            // popular demos and games have taken this approach, and it does hide some artifacts.
            [Tooltip("How the reflections are blended into the render.")]
            public SSRReflectionBlendType blendType;

            [Tooltip("Half resolution SSRR is much faster, but less accurate.")]
            public SSRResolution reflectionQuality;

            [Tooltip("Maximum reflection distance in world units.")]
            [Range(0.1f, 300.0f)]
            public float maxDistance;

            /// REFLECTIONS
            [Tooltip("Max raytracing length.")]
            [Range(16, 1024)]
            public int iterationCount;

            [Tooltip("Log base 2 of ray tracing coarse step size. Higher traces farther, lower gives better quality silhouettes.")]
            [Range(1, 16)]
            public int stepSize;

            [Tooltip("Typical thickness of columns, walls, furniture, and other objects that reflection rays might pass behind.")]
            [Range(0.01f, 10.0f)]
            public float widthModifier;

            [Tooltip("Blurriness of reflections.")]
            [Range(0.1f, 8.0f)]
            public float reflectionBlur;

            [Tooltip("Disable for a performance gain in scenes where most glossy objects are horizontal, like floors, water, and tables. Leave on for scenes with glossy vertical objects.")]
            public bool reflectBackfaces;
        }

        [Serializable]
        public struct ScreenEdgeMask
        {
            [Tooltip("Higher = fade out SSRR near the edge of the screen so that reflections don't pop under camera motion.")]
            [Range(0.0f, 1.0f)]
            public float intensity;
        }

        [Serializable]
        public struct Settings
        {
            public ReflectionSettings reflection;
            public IntensitySettings intensity;
            public ScreenEdgeMask screenEdgeMask;

            public static Settings defaultSettings
            {
                get
                {
                    return new Settings
                    {
                        reflection = new ReflectionSettings
                        {
                            blendType = SSRReflectionBlendType.PhysicallyBased,
                            reflectionQuality = SSRResolution.Low,
                            maxDistance = 100f,
                            iterationCount = 256,
                            stepSize = 3,
                            widthModifier = 0.5f,
                            reflectionBlur = 1f,
                            reflectBackfaces = false
                        },

                        intensity = new IntensitySettings
                        {
                            reflectionMultiplier = 1f,
                            fadeDistance = 100f,

                            fresnelFade = 1f,
                            fresnelFadePower = 1f,
                        },

                        screenEdgeMask = new ScreenEdgeMask
                        {
                            intensity = 0.03f
                        }
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
