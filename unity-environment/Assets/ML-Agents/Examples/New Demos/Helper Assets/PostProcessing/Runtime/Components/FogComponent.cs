using UnityEngine.Rendering;

namespace UnityEngine.PostProcessing
{
    public sealed class FogComponent : PostProcessingComponentCommandBuffer<FogModel>
    {
        static class Uniforms
        {
            internal static readonly int _FogColor = Shader.PropertyToID("_FogColor");
            internal static readonly int _Density  = Shader.PropertyToID("_Density");
            internal static readonly int _Start    = Shader.PropertyToID("_Start");
            internal static readonly int _End      = Shader.PropertyToID("_End");
            internal static readonly int _TempRT   = Shader.PropertyToID("_TempRT");
        }

        const string k_ShaderString = "Hidden/Post FX/Fog";

        public override bool active
        {
            get
            {
                return model.enabled
                       && context.isGBufferAvailable // In forward fog is already done at shader level
                       && RenderSettings.fog
                       && !context.interrupted;
            }
        }

        public override string GetName()
        {
            return "Fog";
        }

        public override DepthTextureMode GetCameraFlags()
        {
            return DepthTextureMode.Depth;
        }

        public override CameraEvent GetCameraEvent()
        {
            return CameraEvent.AfterImageEffectsOpaque;
        }

        public override void PopulateCommandBuffer(CommandBuffer cb)
        {
            var settings = model.settings;

            var material = context.materialFactory.Get(k_ShaderString);
            material.shaderKeywords = null;
            var fogColor = GraphicsUtils.isLinearColorSpace ? RenderSettings.fogColor.linear : RenderSettings.fogColor;
            material.SetColor(Uniforms._FogColor, fogColor);
            material.SetFloat(Uniforms._Density, RenderSettings.fogDensity);
            material.SetFloat(Uniforms._Start, RenderSettings.fogStartDistance);
            material.SetFloat(Uniforms._End, RenderSettings.fogEndDistance);

            switch (RenderSettings.fogMode)
            {
                case FogMode.Linear:
                    material.EnableKeyword("FOG_LINEAR");
                    break;
                case FogMode.Exponential:
                    material.EnableKeyword("FOG_EXP");
                    break;
                case FogMode.ExponentialSquared:
                    material.EnableKeyword("FOG_EXP2");
                    break;
            }

            var fbFormat = context.isHdr
                ? RenderTextureFormat.DefaultHDR
                : RenderTextureFormat.Default;

            cb.GetTemporaryRT(Uniforms._TempRT, context.width, context.height, 24, FilterMode.Bilinear, fbFormat);
            cb.Blit(BuiltinRenderTextureType.CameraTarget, Uniforms._TempRT);
            cb.Blit(Uniforms._TempRT, BuiltinRenderTextureType.CameraTarget, material, settings.excludeSkybox ? 1 : 0);
            cb.ReleaseTemporaryRT(Uniforms._TempRT);
        }
    }
}
