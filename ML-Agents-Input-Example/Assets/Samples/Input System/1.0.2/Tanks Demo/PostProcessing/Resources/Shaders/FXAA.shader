Shader "Hidden/Post FX/FXAA"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }

    CGINCLUDE

        #include "UnityCG.cginc"
        #include "Common.cginc"
        #include "UberSecondPass.cginc"
        #pragma multi_compile __ GRAIN
        #pragma multi_compile __ DITHERING

        #if defined(SHADER_API_PS3)
            #define FXAA_PS3 1

            // Shaves off 2 cycles from the shader
            #define FXAA_EARLY_EXIT 0
        #elif defined(SHADER_API_XBOX360)
            #define FXAA_360 1

            // Shaves off 10ms from the shader's execution time
            #define FXAA_EARLY_EXIT 1
        #else
            #define FXAA_PC 1
        #endif

        #define FXAA_HLSL_3 1
        #define FXAA_QUALITY__PRESET 39

        #define FXAA_GREEN_AS_LUMA 1

        #pragma target 3.0
        #include "FXAA3.cginc"

        float3 _QualitySettings;
        float4 _ConsoleSettings;

        half4 Frag(VaryingsDefault i) : SV_Target
        {
            const float4 consoleUV = i.uv.xyxy + 0.5 * float4(-_MainTex_TexelSize.xy, _MainTex_TexelSize.xy);
            const float4 consoleSubpixelFrame = _ConsoleSettings.x * float4(-1.0, -1.0, 1.0, 1.0) *
                _MainTex_TexelSize.xyxy;

            const float4 consoleSubpixelFramePS3 = float4(-2.0, -2.0, 2.0, 2.0) * _MainTex_TexelSize.xyxy;
            const float4 consoleSubpixelFrameXBOX = float4(8.0, 8.0, -4.0, -4.0) * _MainTex_TexelSize.xyxy;

        #if defined(SHADER_API_XBOX360)
            const float4 consoleConstants = float4(1.0, -1.0, 0.25, -0.25);
        #else
            const float4 consoleConstants = float4(0.0, 0.0, 0.0, 0.0);
        #endif

            half4 color = FxaaPixelShader(
                UnityStereoScreenSpaceUVAdjust(i.uv, _MainTex_ST),
                UnityStereoScreenSpaceUVAdjust(consoleUV, _MainTex_ST),
                _MainTex, _MainTex, _MainTex, _MainTex_TexelSize.xy,
                consoleSubpixelFrame, consoleSubpixelFramePS3, consoleSubpixelFrameXBOX,
                _QualitySettings.x, _QualitySettings.y, _QualitySettings.z,
                _ConsoleSettings.y, _ConsoleSettings.z, _ConsoleSettings.w, consoleConstants);

            color.rgb = UberSecondPass(color.rgb, i.uv);

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
                #pragma fragment Frag

            ENDCG
        }
    }
}
