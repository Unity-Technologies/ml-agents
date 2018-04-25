Shader "Hidden/Post FX/Ambient Occlusion"
{
    CGINCLUDE

        #pragma target 3.0

    ENDCG

    SubShader
    {
        ZTest Always Cull Off ZWrite Off

        // 0: Occlusion estimation with CameraDepthTexture
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragAO
                #pragma multi_compile FOG_OFF FOG_LINEAR FOG_EXP FOG_EXP2
                #define SOURCE_DEPTH
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 1: Occlusion estimation with CameraDepthNormalsTexture
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragAO
                #pragma multi_compile FOG_OFF FOG_LINEAR FOG_EXP FOG_EXP2
                #define SOURCE_DEPTHNORMALS
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 2: Occlusion estimation with G-Buffer
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragAO
                #pragma multi_compile FOG_OFF FOG_LINEAR FOG_EXP FOG_EXP2
                #define SOURCE_GBUFFER
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 3: Separable blur (horizontal pass) with CameraDepthNormalsTexture
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragBlur
                #define SOURCE_DEPTHNORMALS
                #define BLUR_HORIZONTAL
                #define BLUR_SAMPLE_CENTER_NORMAL
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 4: Separable blur (horizontal pass) with G-Buffer
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragBlur
                #define SOURCE_GBUFFER
                #define BLUR_HORIZONTAL
                #define BLUR_SAMPLE_CENTER_NORMAL
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 5: Separable blur (vertical pass)
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragBlur
                #define BLUR_VERTICAL
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 6: Final composition
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragComposition
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 7: Final composition (ambient only mode)
        Pass
        {
            Blend Zero OneMinusSrcColor, Zero OneMinusSrcAlpha

            CGPROGRAM
                #pragma vertex VertCompositionGBuffer
                #pragma fragment FragCompositionGBuffer
                #include "AmbientOcclusion.cginc"
            ENDCG
        }

        // 8: Debug visualization
        Pass
        {
            CGPROGRAM
                #pragma vertex VertMultitex
                #pragma fragment FragComposition
                #define DEBUG_COMPOSITION
                #include "AmbientOcclusion.cginc"
            ENDCG
        }
    }
}
