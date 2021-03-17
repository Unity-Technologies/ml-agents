Shader "Hidden/Post FX/Lut Generator"
{
    CGINCLUDE

        #pragma target 3.0
        #pragma multi_compile __ TONEMAPPING_NEUTRAL TONEMAPPING_FILMIC

        #include "UnityCG.cginc"
        #include "ACES.cginc"
        #include "Common.cginc"
        #include "ColorGrading.cginc"
        #include "Tonemapping.cginc"

        half3 _Balance;

        half3 _Lift;
        half3 _InvGamma;
        half3 _Gain;

        half3 _Offset;
        half3 _Power;
        half3 _Slope;

        half _HueShift;
        half _Saturation;
        half _Contrast;

        half3 _ChannelMixerRed;
        half3 _ChannelMixerGreen;
        half3 _ChannelMixerBlue;

        half4 _NeutralTonemapperParams1;
        half4 _NeutralTonemapperParams2;

        sampler2D _Curves;

        half4 _LutParams;

        half3 ColorGrade(half3 color)
        {
            half3 aces = unity_to_ACES(color);

            // ACEScc (log) space
            half3 acescc = ACES_to_ACEScc(aces);

            acescc = OffsetPowerSlope(acescc, _Offset, _Power, _Slope);

            half2 hs = RgbToHsv(acescc).xy;
            half satMultiplier = SecondaryHueSat(hs.x, _Curves);
            satMultiplier *= SecondarySatSat(hs.y, _Curves);
            satMultiplier *= SecondaryLumSat(AcesLuminance(acescc), _Curves);

            acescc = Saturation(acescc, _Saturation * satMultiplier);
            acescc = ContrastLog(acescc, _Contrast);

            aces = ACEScc_to_ACES(acescc);

            // ACEScg (linear) space
            half3 acescg = ACES_to_ACEScg(aces);

            acescg = WhiteBalance(acescg, _Balance);
            acescg = LiftGammaGain(acescg, _Lift, _InvGamma, _Gain);

            half3 hsv = RgbToHsv(acescg);
            hsv.x = SecondaryHueHue(hsv.x + _HueShift, _Curves);
            acescg = HsvToRgb(hsv);

            acescg = ChannelMixer(acescg, _ChannelMixerRed, _ChannelMixerGreen, _ChannelMixerBlue);

        #if TONEMAPPING_FILMIC

            aces = ACEScg_to_ACES(acescg);
            color = FilmicTonemap(aces);

        #elif TONEMAPPING_NEUTRAL

            color = ACEScg_to_unity(acescg);
            color = NeutralTonemap(color, _NeutralTonemapperParams1, _NeutralTonemapperParams2);

        #else

            color = ACEScg_to_unity(acescg);

        #endif

            // YRGB curves (done in linear/LDR for now)
            color = YrgbCurve(color, _Curves);

            return color;
        }

        half4 FragCreateLut(VaryingsDefault i) : SV_Target
        {
            // 2D strip lut
            half2 uv = i.uv - _LutParams.yz;
            half3 color;
            color.r = frac(uv.x * _LutParams.x);
            color.b = uv.x - color.r / _LutParams.x;
            color.g = uv.y;

            // Lut is in LogC
            half3 colorLogC = color * _LutParams.w;

            // Switch back to unity linear and color grade
            half3 colorLinear = LogCToLinear(colorLogC);
            half3 graded = ColorGrade(colorLinear);

            return half4(graded, 1.0);
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0)
        Pass
        {
            CGPROGRAM

                #pragma vertex VertDefault
                #pragma fragment FragCreateLut

            ENDCG
        }
    }
}
