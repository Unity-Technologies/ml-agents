Shader "Hidden/Post FX/Monitors/Histogram Render"
{
    SubShader
    {
        ZTest Always Cull Off ZWrite Off
        Fog { Mode off }

        CGINCLUDE

            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma target 5.0
            #include "UnityCG.cginc"

            StructuredBuffer<uint4> _Histogram;
            float2 _Size;
            uint _Channel;
            float4 _ColorR;
            float4 _ColorG;
            float4 _ColorB;
            float4 _ColorL;

            float4 FragSingleChannel(v2f_img i) : SV_Target
            {
                const float4 COLORS[4] = { _ColorR, _ColorG, _ColorB, _ColorL };

                float remapI = i.uv.x * 255.0;
                uint index = floor(remapI);
                float delta = frac(remapI);
                float v1 = _Histogram[index][_Channel];
                float v2 = _Histogram[min(index + 1, 255)][_Channel];
                float h = v1 * (1.0 - delta) + v2 * delta;
                uint y = (uint)round(i.uv.y * _Size.y);

                float4 color = float4(0.1, 0.1, 0.1, 1.0);
                float fill = step(y, h);
                color = lerp(color, COLORS[_Channel], fill);
                return color;
            }

            float4 FragRgbMerged(v2f_img i) : SV_Target
            {
                const float4 COLORS[3] = { _ColorR, _ColorG, _ColorB };

                float4 targetColor = float4(0.1, 0.1, 0.1, 1.0);
                float4 emptyColor = float4(0.0, 0.0, 0.0, 1.0);

                float remapI = i.uv.x * 255.0;
                uint index = floor(remapI);
                float delta = frac(remapI);

                for (int j = 0; j < 3; j++)
                {
                    float v1 = _Histogram[index][j];
                    float v2 = _Histogram[min(index + 1, 255)][j];
                    float h = v1 * (1.0 - delta) + v2 * delta;
                    uint y = (uint)round(i.uv.y * _Size.y);
                    float fill = step(y, h);
                    float4 color = lerp(emptyColor, COLORS[j], fill);
                    targetColor += color;
                }

                return saturate(targetColor);
            }

            float4 FragRgbSplitted(v2f_img i) : SV_Target
            {
                const float4 COLORS[3] = {_ColorR, _ColorG, _ColorB};

                const float limitB = round(_Size.y / 3.0);
                const float limitG = limitB * 2;

                float4 color = float4(0.1, 0.1, 0.1, 1.0);
                uint channel;
                float offset;

                if (i.pos.y < limitB)
                {
                    channel = 2;
                    offset = 0.0;
                }
                else if (i.pos.y < limitG)
                {
                    channel = 1;
                    offset = limitB;
                }
                else
                {
                    channel = 0;
                    offset = limitG;
                }

                float remapI = i.uv.x * 255.0;
                uint index = floor(remapI);
                float delta = frac(remapI);
                float v1 = offset + _Histogram[index][channel] / 3.0;
                float v2 = offset + _Histogram[min(index + 1, 255)][channel] / 3.0;
                float h = v1 * (1.0 - delta) + v2 * delta;
                uint y = (uint)round(i.uv.y * _Size.y);

                float fill = step(y, h);
                color = lerp(color, COLORS[channel], fill);
                return color;
            }

        ENDCG

        // (0) Channel
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragSingleChannel

            ENDCG
        }

        // (1) RGB merged
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragRgbMerged

            ENDCG
        }

        // (2) RGB splitted
        Pass
        {
            CGPROGRAM

                #pragma vertex vert_img
                #pragma fragment FragRgbSplitted

            ENDCG
        }
    }
    FallBack off
}
