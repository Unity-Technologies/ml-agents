Shader "Hidden/Post FX/Monitors/Waveform Render"
{
    SubShader
    {
        ZTest Always Cull Off ZWrite Off
        Fog { Mode off }

        CGINCLUDE

            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma target 5.0
            #include "UnityCG.cginc"

            StructuredBuffer<uint4> _Waveform;
            float2 _Size;
            float4 _Channels;
            float _Exposure;

            float3 Tonemap(float3 x, float exposure)
            {
                const float a = 6.2;
                const float b = 0.5;
                const float c = 1.7;
                const float d = 0.06;
                x *= exposure;
                x = max((0.0).xxx, x - (0.004).xxx);
                x = (x * (a * x + b)) / (x * (a * x + c) + d);
                return x * x;
            }

            float4 FragWaveform(v2f_img i) : SV_Target
            {
                const float3 red = float3(1.4, 0.03, 0.02);
                const float3 green = float3(0.02, 1.1, 0.05);
                const float3 blue = float3(0.0, 0.25, 1.5);
                float3 color = float3(0.0, 0.0, 0.0);

                uint2 uvI = i.pos.xy;
                float4 w = _Waveform[uvI.y + uvI.x * _Size.y]; // Waveform data is stored in columns instead of rows

                color += red * w.r * _Channels.r;
                color += green * w.g * _Channels.g;
                color += blue * w.b * _Channels.b;
                color += w.aaa * _Channels.a * 1.5;
                color = Tonemap(color, _Exposure);
                color += (0.1).xxx;

                return float4(saturate(color), 1.0);
            }

        ENDCG

        // (0)
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragWaveform

            ENDCG
        }
    }
    FallBack off
}
