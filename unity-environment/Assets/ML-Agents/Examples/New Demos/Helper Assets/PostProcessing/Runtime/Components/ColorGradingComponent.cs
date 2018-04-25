namespace UnityEngine.PostProcessing
{
    using DebugMode = BuiltinDebugViewsModel.Mode;

    public sealed class ColorGradingComponent : PostProcessingComponentRenderTexture<ColorGradingModel>
    {
        static class Uniforms
        {
            internal static readonly int _LutParams                = Shader.PropertyToID("_LutParams");
            internal static readonly int _NeutralTonemapperParams1 = Shader.PropertyToID("_NeutralTonemapperParams1");
            internal static readonly int _NeutralTonemapperParams2 = Shader.PropertyToID("_NeutralTonemapperParams2");
            internal static readonly int _HueShift                 = Shader.PropertyToID("_HueShift");
            internal static readonly int _Saturation               = Shader.PropertyToID("_Saturation");
            internal static readonly int _Contrast                 = Shader.PropertyToID("_Contrast");
            internal static readonly int _Balance                  = Shader.PropertyToID("_Balance");
            internal static readonly int _Lift                     = Shader.PropertyToID("_Lift");
            internal static readonly int _InvGamma                 = Shader.PropertyToID("_InvGamma");
            internal static readonly int _Gain                     = Shader.PropertyToID("_Gain");
            internal static readonly int _Slope                    = Shader.PropertyToID("_Slope");
            internal static readonly int _Power                    = Shader.PropertyToID("_Power");
            internal static readonly int _Offset                   = Shader.PropertyToID("_Offset");
            internal static readonly int _ChannelMixerRed          = Shader.PropertyToID("_ChannelMixerRed");
            internal static readonly int _ChannelMixerGreen        = Shader.PropertyToID("_ChannelMixerGreen");
            internal static readonly int _ChannelMixerBlue         = Shader.PropertyToID("_ChannelMixerBlue");
            internal static readonly int _Curves                   = Shader.PropertyToID("_Curves");
            internal static readonly int _LogLut                   = Shader.PropertyToID("_LogLut");
            internal static readonly int _LogLut_Params            = Shader.PropertyToID("_LogLut_Params");
            internal static readonly int _ExposureEV               = Shader.PropertyToID("_ExposureEV");
        }

        const int k_InternalLogLutSize = 32;
        const int k_CurvePrecision = 128;
        const float k_CurveStep = 1f / k_CurvePrecision;

        Texture2D m_GradingCurves;
        Color[] m_pixels = new Color[k_CurvePrecision * 2];

        public override bool active
        {
            get
            {
                return model.enabled
                       && !context.interrupted;
            }
        }

        // An analytical model of chromaticity of the standard illuminant, by Judd et al.
        // http://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_series_D
        // Slightly modifed to adjust it with the D65 white point (x=0.31271, y=0.32902).
        float StandardIlluminantY(float x)
        {
            return 2.87f * x - 3f * x * x - 0.27509507f;
        }

        // CIE xy chromaticity to CAT02 LMS.
        // http://en.wikipedia.org/wiki/LMS_color_space#CAT02
        Vector3 CIExyToLMS(float x, float y)
        {
            float Y = 1f;
            float X = Y * x / y;
            float Z = Y * (1f - x - y) / y;

            float L =  0.7328f * X + 0.4296f * Y - 0.1624f * Z;
            float M = -0.7036f * X + 1.6975f * Y + 0.0061f * Z;
            float S =  0.0030f * X + 0.0136f * Y + 0.9834f * Z;

            return new Vector3(L, M, S);
        }

        Vector3 CalculateColorBalance(float temperature, float tint)
        {
            // Range ~[-1.8;1.8] ; using higher ranges is unsafe
            float t1 = temperature / 55f;
            float t2 = tint / 55f;

            // Get the CIE xy chromaticity of the reference white point.
            // Note: 0.31271 = x value on the D65 white point
            float x = 0.31271f - t1 * (t1 < 0f ? 0.1f : 0.05f);
            float y = StandardIlluminantY(x) + t2 * 0.05f;

            // Calculate the coefficients in the LMS space.
            var w1 = new Vector3(0.949237f, 1.03542f, 1.08728f); // D65 white point
            var w2 = CIExyToLMS(x, y);
            return new Vector3(w1.x / w2.x, w1.y / w2.y, w1.z / w2.z);
        }

        static Color NormalizeColor(Color c)
        {
            float sum = (c.r + c.g + c.b) / 3f;

            if (Mathf.Approximately(sum, 0f))
                return new Color(1f, 1f, 1f, c.a);

            return new Color
                   {
                       r = c.r / sum,
                       g = c.g / sum,
                       b = c.b / sum,
                       a = c.a
                   };
        }

        static Vector3 ClampVector(Vector3 v, float min, float max)
        {
            return new Vector3(
                Mathf.Clamp(v.x, min, max),
                Mathf.Clamp(v.y, min, max),
                Mathf.Clamp(v.z, min, max)
                );
        }

        public static Vector3 GetLiftValue(Color lift)
        {
            const float kLiftScale = 0.1f;

            var nLift = NormalizeColor(lift);
            float avgLift = (nLift.r + nLift.g + nLift.b) / 3f;

            // Getting some artifacts when going into the negatives using a very low offset (lift.a) with non ACES-tonemapping
            float liftR = (nLift.r - avgLift) * kLiftScale + lift.a;
            float liftG = (nLift.g - avgLift) * kLiftScale + lift.a;
            float liftB = (nLift.b - avgLift) * kLiftScale + lift.a;

            return ClampVector(new Vector3(liftR, liftG, liftB), -1f, 1f);
        }

        public static Vector3 GetGammaValue(Color gamma)
        {
            const float kGammaScale = 0.5f;
            const float kMinGamma = 0.01f;

            var nGamma = NormalizeColor(gamma);
            float avgGamma = (nGamma.r + nGamma.g + nGamma.b) / 3f;

            gamma.a *= gamma.a < 0f ? 0.8f : 5f;
            float gammaR = Mathf.Pow(2f, (nGamma.r - avgGamma) * kGammaScale) + gamma.a;
            float gammaG = Mathf.Pow(2f, (nGamma.g - avgGamma) * kGammaScale) + gamma.a;
            float gammaB = Mathf.Pow(2f, (nGamma.b - avgGamma) * kGammaScale) + gamma.a;

            float invGammaR = 1f / Mathf.Max(kMinGamma, gammaR);
            float invGammaG = 1f / Mathf.Max(kMinGamma, gammaG);
            float invGammaB = 1f / Mathf.Max(kMinGamma, gammaB);

            return ClampVector(new Vector3(invGammaR, invGammaG, invGammaB), 0f, 5f);
        }

        public static Vector3 GetGainValue(Color gain)
        {
            const float kGainScale = 0.5f;

            var nGain = NormalizeColor(gain);
            float avgGain = (nGain.r + nGain.g + nGain.b) / 3f;

            gain.a *= gain.a > 0f ? 3f : 1f;
            float gainR = Mathf.Pow(2f, (nGain.r - avgGain) * kGainScale) + gain.a;
            float gainG = Mathf.Pow(2f, (nGain.g - avgGain) * kGainScale) + gain.a;
            float gainB = Mathf.Pow(2f, (nGain.b - avgGain) * kGainScale) + gain.a;

            return ClampVector(new Vector3(gainR, gainG, gainB), 0f, 4f);
        }

        public static void CalculateLiftGammaGain(Color lift, Color gamma, Color gain, out Vector3 outLift, out Vector3 outGamma, out Vector3 outGain)
        {
            outLift = GetLiftValue(lift);
            outGamma = GetGammaValue(gamma);
            outGain = GetGainValue(gain);
        }

        public static Vector3 GetSlopeValue(Color slope)
        {
            const float kSlopeScale = 0.1f;

            var nSlope = NormalizeColor(slope);
            float avgSlope = (nSlope.r + nSlope.g + nSlope.b) / 3f;

            slope.a *= 0.5f;
            float slopeR = (nSlope.r - avgSlope) * kSlopeScale + slope.a + 1f;
            float slopeG = (nSlope.g - avgSlope) * kSlopeScale + slope.a + 1f;
            float slopeB = (nSlope.b - avgSlope) * kSlopeScale + slope.a + 1f;

            return ClampVector(new Vector3(slopeR, slopeG, slopeB), 0f, 2f);
        }

        public static Vector3 GetPowerValue(Color power)
        {
            const float kPowerScale = 0.1f;
            const float minPower = 0.01f;

            var nPower = NormalizeColor(power);
            float avgPower = (nPower.r + nPower.g + nPower.b) / 3f;

            power.a *= 0.5f;
            float powerR = (nPower.r - avgPower) * kPowerScale + power.a + 1f;
            float powerG = (nPower.g - avgPower) * kPowerScale + power.a + 1f;
            float powerB = (nPower.b - avgPower) * kPowerScale + power.a + 1f;

            float invPowerR = 1f / Mathf.Max(minPower, powerR);
            float invPowerG = 1f / Mathf.Max(minPower, powerG);
            float invPowerB = 1f / Mathf.Max(minPower, powerB);

            return ClampVector(new Vector3(invPowerR, invPowerG, invPowerB), 0.5f, 2.5f);
        }

        public static Vector3 GetOffsetValue(Color offset)
        {
            const float kOffsetScale = 0.05f;

            var nOffset = NormalizeColor(offset);
            float avgOffset = (nOffset.r + nOffset.g + nOffset.b) / 3f;

            offset.a *= 0.5f;
            float offsetR = (nOffset.r - avgOffset) * kOffsetScale + offset.a;
            float offsetG = (nOffset.g - avgOffset) * kOffsetScale + offset.a;
            float offsetB = (nOffset.b - avgOffset) * kOffsetScale + offset.a;

            return ClampVector(new Vector3(offsetR, offsetG, offsetB), -0.8f, 0.8f);
        }

        public static void CalculateSlopePowerOffset(Color slope, Color power, Color offset, out Vector3 outSlope, out Vector3 outPower, out Vector3 outOffset)
        {
            outSlope = GetSlopeValue(slope);
            outPower = GetPowerValue(power);
            outOffset = GetOffsetValue(offset);
        }

        TextureFormat GetCurveFormat()
        {
            if (SystemInfo.SupportsTextureFormat(TextureFormat.RGBAHalf))
                return TextureFormat.RGBAHalf;

            return TextureFormat.RGBA32;
        }

        Texture2D GetCurveTexture()
        {
            if (m_GradingCurves == null)
            {
                m_GradingCurves = new Texture2D(k_CurvePrecision, 2, GetCurveFormat(), false, true)
                {
                    name = "Internal Curves Texture",
                    hideFlags = HideFlags.DontSave,
                    anisoLevel = 0,
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear
                };
            }

            var curves = model.settings.curves;
            curves.hueVShue.Cache();
            curves.hueVSsat.Cache();

            for (int i = 0; i < k_CurvePrecision; i++)
            {
                float t = i * k_CurveStep;

                // HSL
                float x = curves.hueVShue.Evaluate(t);
                float y = curves.hueVSsat.Evaluate(t);
                float z = curves.satVSsat.Evaluate(t);
                float w = curves.lumVSsat.Evaluate(t);
                m_pixels[i] = new Color(x, y, z, w);

                // YRGB
                float m = curves.master.Evaluate(t);
                float r = curves.red.Evaluate(t);
                float g = curves.green.Evaluate(t);
                float b = curves.blue.Evaluate(t);
                m_pixels[i + k_CurvePrecision] = new Color(r, g, b, m);
            }

            m_GradingCurves.SetPixels(m_pixels);
            m_GradingCurves.Apply(false, false);

            return m_GradingCurves;
        }

        bool IsLogLutValid(RenderTexture lut)
        {
            return lut != null && lut.IsCreated() && lut.height == k_InternalLogLutSize;
        }

        RenderTextureFormat GetLutFormat()
        {
            if (SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.ARGBHalf))
                return RenderTextureFormat.ARGBHalf;

            return RenderTextureFormat.ARGB32;
        }

        void GenerateLut()
        {
            var settings = model.settings;

            if (!IsLogLutValid(model.bakedLut))
            {
                GraphicsUtils.Destroy(model.bakedLut);

                model.bakedLut = new RenderTexture(k_InternalLogLutSize * k_InternalLogLutSize, k_InternalLogLutSize, 0, GetLutFormat())
                {
                    name = "Color Grading Log LUT",
                    hideFlags = HideFlags.DontSave,
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp,
                    anisoLevel = 0
                };
            }

            var lutMaterial = context.materialFactory.Get("Hidden/Post FX/Lut Generator");
            lutMaterial.SetVector(Uniforms._LutParams, new Vector4(
                    k_InternalLogLutSize,
                    0.5f / (k_InternalLogLutSize * k_InternalLogLutSize),
                    0.5f / k_InternalLogLutSize,
                    k_InternalLogLutSize / (k_InternalLogLutSize - 1f))
                );

            // Tonemapping
            lutMaterial.shaderKeywords = null;

            var tonemapping = settings.tonemapping;
            switch (tonemapping.tonemapper)
            {
                case ColorGradingModel.Tonemapper.Neutral:
                {
                    lutMaterial.EnableKeyword("TONEMAPPING_NEUTRAL");

                    const float scaleFactor = 20f;
                    const float scaleFactorHalf = scaleFactor * 0.5f;

                    float inBlack = tonemapping.neutralBlackIn * scaleFactor + 1f;
                    float outBlack = tonemapping.neutralBlackOut * scaleFactorHalf + 1f;
                    float inWhite = tonemapping.neutralWhiteIn / scaleFactor;
                    float outWhite = 1f - tonemapping.neutralWhiteOut / scaleFactor;
                    float blackRatio = inBlack / outBlack;
                    float whiteRatio = inWhite / outWhite;

                    const float a = 0.2f;
                    float b = Mathf.Max(0f, Mathf.LerpUnclamped(0.57f, 0.37f, blackRatio));
                    float c = Mathf.LerpUnclamped(0.01f, 0.24f, whiteRatio);
                    float d = Mathf.Max(0f, Mathf.LerpUnclamped(0.02f, 0.20f, blackRatio));
                    const float e = 0.02f;
                    const float f = 0.30f;

                    lutMaterial.SetVector(Uniforms._NeutralTonemapperParams1, new Vector4(a, b, c, d));
                    lutMaterial.SetVector(Uniforms._NeutralTonemapperParams2, new Vector4(e, f, tonemapping.neutralWhiteLevel, tonemapping.neutralWhiteClip / scaleFactorHalf));
                    break;
                }

                case ColorGradingModel.Tonemapper.ACES:
                {
                    lutMaterial.EnableKeyword("TONEMAPPING_FILMIC");
                    break;
                }
            }

            // Color balance & basic grading settings
            lutMaterial.SetFloat(Uniforms._HueShift, settings.basic.hueShift / 360f);
            lutMaterial.SetFloat(Uniforms._Saturation, settings.basic.saturation);
            lutMaterial.SetFloat(Uniforms._Contrast, settings.basic.contrast);
            lutMaterial.SetVector(Uniforms._Balance, CalculateColorBalance(settings.basic.temperature, settings.basic.tint));

            // Lift / Gamma / Gain
            Vector3 lift, gamma, gain;
            CalculateLiftGammaGain(
                settings.colorWheels.linear.lift,
                settings.colorWheels.linear.gamma,
                settings.colorWheels.linear.gain,
                out lift, out gamma, out gain
                );

            lutMaterial.SetVector(Uniforms._Lift, lift);
            lutMaterial.SetVector(Uniforms._InvGamma, gamma);
            lutMaterial.SetVector(Uniforms._Gain, gain);

            // Slope / Power / Offset
            Vector3 slope, power, offset;
            CalculateSlopePowerOffset(
                settings.colorWheels.log.slope,
                settings.colorWheels.log.power,
                settings.colorWheels.log.offset,
                out slope, out power, out offset
                );

            lutMaterial.SetVector(Uniforms._Slope, slope);
            lutMaterial.SetVector(Uniforms._Power, power);
            lutMaterial.SetVector(Uniforms._Offset, offset);

            // Channel mixer
            lutMaterial.SetVector(Uniforms._ChannelMixerRed, settings.channelMixer.red);
            lutMaterial.SetVector(Uniforms._ChannelMixerGreen, settings.channelMixer.green);
            lutMaterial.SetVector(Uniforms._ChannelMixerBlue, settings.channelMixer.blue);

            // Selective grading & YRGB curves
            lutMaterial.SetTexture(Uniforms._Curves, GetCurveTexture());

            // Generate the lut
            Graphics.Blit(null, model.bakedLut, lutMaterial, 0);
        }

        public override void Prepare(Material uberMaterial)
        {
            if (model.isDirty || !IsLogLutValid(model.bakedLut))
            {
                GenerateLut();
                model.isDirty = false;
            }

            uberMaterial.EnableKeyword(
                context.profile.debugViews.IsModeActive(DebugMode.PreGradingLog)
                ? "COLOR_GRADING_LOG_VIEW"
                : "COLOR_GRADING"
                );

            var bakedLut = model.bakedLut;
            uberMaterial.SetTexture(Uniforms._LogLut, bakedLut);
            uberMaterial.SetVector(Uniforms._LogLut_Params, new Vector3(1f / bakedLut.width, 1f / bakedLut.height, bakedLut.height - 1f));

            float ev = Mathf.Exp(model.settings.basic.postExposure * 0.69314718055994530941723212145818f);
            uberMaterial.SetFloat(Uniforms._ExposureEV, ev);
        }

        public void OnGUI()
        {
            var bakedLut = model.bakedLut;
            var rect = new Rect(context.viewport.x * Screen.width + 8f, 8f, bakedLut.width, bakedLut.height);
            GUI.DrawTexture(rect, bakedLut);
        }

        public override void OnDisable()
        {
            GraphicsUtils.Destroy(m_GradingCurves);
            GraphicsUtils.Destroy(model.bakedLut);
            m_GradingCurves = null;
            model.bakedLut = null;
        }
    }
}
