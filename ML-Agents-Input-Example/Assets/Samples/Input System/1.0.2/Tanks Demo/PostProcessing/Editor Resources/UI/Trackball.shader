Shader "Hidden/Post FX/UI/Trackball"
{
    CGINCLUDE

        #include "UnityCG.cginc"

        #define PI 3.14159265359
        #define PI2 6.28318530718

        float _Offset;
        float _DisabledState;
        float2 _Resolution; // x: size, y: size / 2

        float3 HsvToRgb(float3 c)
        {
            float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
        }

        float4 CreateWheel(v2f_img i, float crossColor, float offsetColor)
        {
            const float kHueOuterRadius = 0.45;
            const float kHueInnerRadius = 0.38;
            const float kLumOuterRadius = 0.495;
            const float kLumInnerRadius = 0.48;

            float4 color = (0.0).xxxx;
            float2 uvc = i.uv - (0.5).xx;
            float dist = sqrt(dot(uvc, uvc));
            float delta = fwidth(dist);
            float angle = atan2(uvc.x, uvc.y);

            // Cross
            {
                float radius = (0.5 - kHueInnerRadius) * _Resolution.x + 1.0;
                float2 pixel = (_Resolution.xx - 1.0) * i.uv + 1.0;

                float vline = step(floor(fmod(pixel.x, _Resolution.y)), 0.0);
                vline *= step(radius, pixel.y) * step(pixel.y, _Resolution.x - radius);

                float hline = step(floor(fmod(pixel.y, _Resolution.y)), 0.0);
                hline *= step(radius, pixel.x) * step(pixel.x, _Resolution.x - radius);

                color += hline.xxxx * (1.0).xxxx;
                color += vline.xxxx * (1.0).xxxx;
                color = saturate(color);
                color *= half4((crossColor).xxx, 0.05);
            }

            // Hue
            {
                float alphaOut = smoothstep(kHueOuterRadius - delta, kHueOuterRadius + delta, dist);
                float alphaIn = smoothstep(kHueInnerRadius - delta, kHueInnerRadius + delta, dist);

                float hue = angle;
                hue = 1.0 - ((hue > 0.0) ? hue : PI2 + hue) / PI2;
                float4 c = float4(HsvToRgb(float3(hue, 1.0, 1.0)), 1.0);
                color += lerp((0.0).xxxx, c, alphaIn - alphaOut);
            }

            // Offset
            {
                float alphaOut = smoothstep(kLumOuterRadius - delta, kLumOuterRadius + delta, dist);
                float alphaIn = smoothstep(kLumInnerRadius - delta, kLumInnerRadius + delta / 2, dist);
                float4 c = float4((offsetColor).xxx, 1.0);

                float a = PI * _Offset;
                if (_Offset >= 0 && angle < a && angle > 0.0)
                    c = float4((1.0).xxx, 0.5);
                else if (angle > a && angle < 0.0)
                    c = float4((1.0).xxx, 0.5);

                color += lerp((0.0).xxxx, c, alphaIn - alphaOut);
            }

            return color * _DisabledState;
        }

        float4 FragTrackballDark(v2f_img i) : SV_Target
        {
            return CreateWheel(i, 1.0, 0.15);
        }

        float4 FragTrackballLight(v2f_img i) : SV_Target
        {
            return CreateWheel(i, 0.0, 0.3);
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0) Dark skin
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragTrackballDark

            ENDCG
        }

        // (1) Light skin
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragTrackballLight

            ENDCG
        }
    }
}
