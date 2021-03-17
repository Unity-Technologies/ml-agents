Shader "Hidden/Post FX/Monitors/Vectorscope Render"
{
    SubShader
    {
        ZTest Always Cull Off ZWrite Off
        Fog { Mode off }

        CGINCLUDE

            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma target 5.0
            #include "UnityCG.cginc"

            StructuredBuffer<uint> _Vectorscope;
            float2 _Size;
            float _Exposure;

            float Tonemap(float x, float exposure)
            {
                const float a = 6.2;
                const float b = 0.5;
                const float c = 1.7;
                const float d = 0.06;
                x *= exposure;
                x = max(0.0, x - 0.004);
                x = (x * (a * x + b)) / (x * (a * x + c) + d);
                return x * x;
            }

            float3 YuvToRgb(float3 c)
            {
                float R = c.x + 0.000 * c.y + 1.403 * c.z;
                float G = c.x - 0.344 * c.y - 0.714 * c.z;
                float B = c.x - 1.773 * c.y + 0.000 * c.z;
                return float3(R, G, B);
            }

            float4 FragBackground(v2f_img i) : SV_Target
            {
                i.uv.x = 1.0 - i.uv.x;
                float2 uv = i.uv - (0.5).xx;
                float3 c = YuvToRgb(float3(0.5, uv.x, uv.y));

                float dist = sqrt(dot(uv, uv));
                float delta = fwidth(dist);
                float alphaOut = 1.0 - smoothstep(0.5 - delta, 0.5 + delta, dist);
                float alphaIn = smoothstep(0.495 - delta, 0.495 + delta, dist);

                uint2 uvI = i.pos.xy;
                uint v = _Vectorscope[uvI.x + uvI.y * _Size.x];
                float vt = saturate(Tonemap(v, _Exposure));

                float4 color = float4(lerp(c, (0.0).xxx, vt), alphaOut);
                color.rgb += alphaIn;
                return color;
            }

            float4 FragNoBackground(v2f_img i) : SV_Target
            {
                i.uv.x = 1.0 - i.uv.x;
                float2 uv = i.uv - (0.5).xx;

                float dist = sqrt(dot(uv, uv));
                float delta = fwidth(dist);
                float alphaOut = 1.0 - smoothstep(0.5 - delta, 0.5 + delta, dist);
                float alphaIn = smoothstep(0.495 - delta, 0.495 + delta, dist);

                uint2 uvI = i.pos.xy;
                uint v = _Vectorscope[uvI.x + uvI.y * _Size.x];
                float vt = saturate(Tonemap(v, _Exposure));

                float4 color = float4((1.0).xxx, vt + alphaIn * alphaOut);
                return color;
            }

        ENDCG

        // (0)
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragBackground

            ENDCG
        }

        // (1)
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragNoBackground

            ENDCG
        }
    }
    FallBack off
}
