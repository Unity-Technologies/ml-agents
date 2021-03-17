Shader "Hidden/Post FX/Monitors/Parade Render"
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
            float4 _Size;
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

            float4 FragParade(v2f_img i) : SV_Target
            {
                const float3 red = float3(1.8, 0.03, 0.02);
                const float3 green = float3(0.02, 1.3, 0.05);
                const float3 blue = float3(0.0, 0.45, 1.75);
                float3 color = float3(0.0, 0.0, 0.0);

                const uint limitR = _Size.x / 3;
                const uint limitG = limitR * 2;

                if (i.pos.x < (float)limitR)
                {
                    uint2 uvI = i.pos.xy;
                    color = _Waveform[uvI.y + uvI.x * _Size.y].r * red;
                }
                else if (i.pos.x < (float)limitG)
                {
                    uint2 uvI = uint2(i.pos.x - limitR, i.pos.y);
                    color = _Waveform[uvI.y + uvI.x * _Size.y].g * green;
                }
                else
                {
                    uint2 uvI = uint2(i.pos.x - limitG, i.pos.y);
                    color = _Waveform[uvI.y + uvI.x * _Size.y].b * blue;
                }

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
                #pragma fragment FragParade

            ENDCG
        }
    }
    FallBack off
}
