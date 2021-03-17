Shader "Hidden/Post FX/Eye Adaptation"
{
    Properties
    {
        _MainTex("Texture", 2D) = "white" {}
    }

    CGINCLUDE

        #pragma target 4.5
        #pragma multi_compile __ AUTO_KEY_VALUE
        #include "UnityCG.cginc"
        #include "Common.cginc"
        #include "EyeAdaptation.cginc"

        // Eye adaptation pass
        float4 _Params; // x: lowPercent, y: highPercent, z: minBrightness, w: maxBrightness
        float2 _Speed; // x: down, y: up
        float4 _ScaleOffsetRes; // x: scale, y: offset, w: histogram pass width, h: histogram pass height
        float _ExposureCompensation;

        StructuredBuffer<uint> _Histogram;

        float GetBinValue(uint index, float maxHistogramValue)
        {
            return float(_Histogram[index]) * maxHistogramValue;
        }

        // Done in the vertex shader
        float FindMaxHistogramValue()
        {
            uint maxValue = 0u;

            for (uint i = 0; i < HISTOGRAM_BINS; i++)
            {
                uint h = _Histogram[i];
                maxValue = max(maxValue, h);
            }

            return float(maxValue);
        }

        void FilterLuminance(uint i, float maxHistogramValue, inout float4 filter)
        {
            float binValue = GetBinValue(i, maxHistogramValue);

            // Filter dark areas
            float offset = min(filter.z, binValue);
            binValue -= offset;
            filter.zw -= offset.xx;

            // Filter highlights
            binValue = min(filter.w, binValue);
            filter.w -= binValue;

            // Luminance at the bin
            float luminance = GetLuminanceFromHistogramBin(float(i) / float(HISTOGRAM_BINS), _ScaleOffsetRes.xy);

            filter.xy += float2(luminance * binValue, binValue);
        }

        float GetAverageLuminance(float maxHistogramValue)
        {
            // Sum of all bins
            uint i;
            float totalSum = 0.0;

            UNITY_LOOP
            for (i = 0; i < HISTOGRAM_BINS; i++)
                totalSum += GetBinValue(i, maxHistogramValue);

            // Skip darker and lighter parts of the histogram to stabilize the auto exposure
            // x: filtered sum
            // y: accumulator
            // zw: fractions
            float4 filter = float4(0.0, 0.0, totalSum * _Params.xy);

            UNITY_LOOP
            for (i = 0; i < HISTOGRAM_BINS; i++)
                FilterLuminance(i, maxHistogramValue, filter);

            // Clamp to user brightness range
            return clamp(filter.x / max(filter.y, EPSILON), _Params.z, _Params.w);
        }

        float GetExposureMultiplier(float avgLuminance)
        {
            avgLuminance = max(EPSILON, avgLuminance);

        #if AUTO_KEY_VALUE
            half keyValue = 1.03 - (2.0 / (2.0 + log2(avgLuminance + 1.0)));
        #else
            half keyValue = _ExposureCompensation;
        #endif

            half exposure = keyValue / avgLuminance;

            return exposure;
        }

        float InterpolateExposure(float newExposure, float oldExposure)
        {
            float delta = newExposure - oldExposure;
            float speed = delta > 0.0 ? _Speed.x : _Speed.y;
            float exposure = oldExposure + delta * (1.0 - exp2(-unity_DeltaTime.x * speed));
            //float exposure = oldExposure + delta * (unity_DeltaTime.x * speed);
            return exposure;
        }

        float4 FragAdaptProgressive(VaryingsDefault i) : SV_Target
        {
            float maxValue = 1.0 / FindMaxHistogramValue();
            float avgLuminance = GetAverageLuminance(maxValue);
            float exposure = GetExposureMultiplier(avgLuminance);
            float prevExposure = tex2D(_MainTex, (0.5).xx);
            exposure = InterpolateExposure(exposure, prevExposure);
            return exposure.xxxx;
        }

        float4 FragAdaptFixed(VaryingsDefault i) : SV_Target
        {
            float maxValue = 1.0 / FindMaxHistogramValue();
            float avgLuminance = GetAverageLuminance(maxValue);
            float exposure = GetExposureMultiplier(avgLuminance);
            return exposure.xxxx;
        }

        // ---- Editor stuff
        int _DebugWidth;

        struct VaryingsEditorHisto
        {
            float4 pos : SV_POSITION;
            float2 uv : TEXCOORD0;
            float maxValue : TEXCOORD1;
            float avgLuminance : TEXCOORD2;
        };

        VaryingsEditorHisto VertEditorHisto(AttributesDefault v)
        {
            VaryingsEditorHisto o;
            o.pos = UnityObjectToClipPos(v.vertex);
            o.uv = v.texcoord.xy;
            o.maxValue = 1.0 / FindMaxHistogramValue();
            o.avgLuminance = GetAverageLuminance(o.maxValue);
            return o;
        }

        float4 FragEditorHisto(VaryingsEditorHisto i) : SV_Target
        {
            const float3 kRangeColor = float3(0.05, 0.4, 0.6);
            const float3 kAvgColor = float3(0.8, 0.3, 0.05);

            float4 color = float4(0.0, 0.0, 0.0, 0.7);

            uint ix = (uint)(round(i.uv.x * HISTOGRAM_BINS));
            float bin = saturate(float(_Histogram[ix]) * i.maxValue);
            float fill = step(i.uv.y, bin);

            // Min / max brightness markers
            float luminanceMin = GetHistogramBinFromLuminance(_Params.z, _ScaleOffsetRes.xy);
            float luminanceMax = GetHistogramBinFromLuminance(_Params.w, _ScaleOffsetRes.xy);

            color.rgb += fill.rrr;

            if (i.uv.x > luminanceMin && i.uv.x < luminanceMax)
            {
                color.rgb = fill.rrr * kRangeColor;
                color.rgb += kRangeColor;
            }

            // Current average luminance marker
            float luminanceAvg = GetHistogramBinFromLuminance(i.avgLuminance, _ScaleOffsetRes.xy);
            float avgPx = luminanceAvg * _DebugWidth;

            if (abs(i.pos.x - avgPx) < 2)
                color.rgb = kAvgColor;

            return color;
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM

                #pragma vertex VertDefault
                #pragma fragment FragAdaptProgressive

            ENDCG
        }

        Pass
        {
            CGPROGRAM

                #pragma vertex VertDefault
                #pragma fragment FragAdaptFixed

            ENDCG
        }

        Pass
        {
            CGPROGRAM

                #pragma vertex VertEditorHisto
                #pragma fragment FragEditorHisto

            ENDCG
        }
    }
}
