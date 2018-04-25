namespace UnityEngine.PostProcessing
{
    public sealed class VignetteComponent : PostProcessingComponentRenderTexture<VignetteModel>
    {
        static class Uniforms
        {
            internal static readonly int _Vignette_Color    = Shader.PropertyToID("_Vignette_Color");
            internal static readonly int _Vignette_Center   = Shader.PropertyToID("_Vignette_Center");
            internal static readonly int _Vignette_Settings = Shader.PropertyToID("_Vignette_Settings");
            internal static readonly int _Vignette_Mask     = Shader.PropertyToID("_Vignette_Mask");
            internal static readonly int _Vignette_Opacity  = Shader.PropertyToID("_Vignette_Opacity");
        }

        public override bool active
        {
            get
            {
                return model.enabled
                       && !context.interrupted;
            }
        }

        public override void Prepare(Material uberMaterial)
        {
            var settings = model.settings;
            uberMaterial.SetColor(Uniforms._Vignette_Color, settings.color);

            if (settings.mode == VignetteModel.Mode.Classic)
            {
                uberMaterial.SetVector(Uniforms._Vignette_Center, settings.center);
                uberMaterial.EnableKeyword("VIGNETTE_CLASSIC");
                float roundness = (1f - settings.roundness) * 6f + settings.roundness;
                uberMaterial.SetVector(Uniforms._Vignette_Settings, new Vector4(settings.intensity * 3f, settings.smoothness * 5f, roundness, settings.rounded ? 1f : 0f));
            }
            else if (settings.mode == VignetteModel.Mode.Masked)
            {
                if (settings.mask != null && settings.opacity > 0f)
                {
                    uberMaterial.EnableKeyword("VIGNETTE_MASKED");
                    uberMaterial.SetTexture(Uniforms._Vignette_Mask, settings.mask);
                    uberMaterial.SetFloat(Uniforms._Vignette_Opacity, settings.opacity);
                }
            }
        }
    }
}
