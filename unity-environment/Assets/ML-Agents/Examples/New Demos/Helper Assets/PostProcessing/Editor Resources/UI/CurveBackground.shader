Shader "Hidden/Post FX/UI/Curve Background"
{
    CGINCLUDE

        #pragma target 3.0
        #include "UnityCG.cginc"

        float _DisabledState;

        float3 HsvToRgb(float3 c)
        {
            float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
        }

        float4 FragHue(v2f_img i) : SV_Target
        {
            float3 hsv = HsvToRgb(float3(i.uv.x, 1.0, 0.2));
            float4 color = float4((0.0).xxx, 1.0);
            color.rgb = lerp(color.rgb, hsv, smoothstep(0.5, 1.1, 1.0 - i.uv.y)) + lerp(color.rgb, hsv, smoothstep(0.5, 1.1, i.uv.y));
            color.rgb += (0.15).xxx;
            return float4(color.rgb, color.a * _DisabledState);
        }

        float4 FragSat(v2f_img i) : SV_Target
        {
            float4 color = float4((0.0).xxx, 1.0);
            float sat = i.uv.x / 2;
            color.rgb += lerp(color.rgb, (sat).xxx, smoothstep(0.5, 1.2, 1.0 - i.uv.y)) + lerp(color.rgb, (sat).xxx, smoothstep(0.5, 1.2, i.uv.y));
            color.rgb += (0.15).xxx;
            return float4(color.rgb, color.a * _DisabledState);
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0) Hue
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragHue

            ENDCG
        }

        // (1) Sat/lum
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragSat

            ENDCG
        }
    }
}
