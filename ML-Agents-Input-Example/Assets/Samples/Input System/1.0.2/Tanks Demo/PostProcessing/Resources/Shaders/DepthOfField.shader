Shader "Hidden/Post FX/Depth Of Field"
{
    Properties
    {
        _MainTex ("", 2D) = "black"
    }

    CGINCLUDE
        #pragma exclude_renderers d3d11_9x
        #pragma target 3.0
    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // (0) Downsampling, prefiltering & CoC
        Pass
        {
            CGPROGRAM
                #pragma multi_compile __ UNITY_COLORSPACE_GAMMA
                #pragma vertex VertDOF
                #pragma fragment FragPrefilter
                #include "DepthOfField.cginc"
            ENDCG
        }

        // (1) Pass 0 + temporal antialiasing
        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragPrefilter
                #define PREFILTER_TAA
                #include "DepthOfField.cginc"
            ENDCG
        }

        // (2-5) Bokeh filter with disk-shaped kernels
        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_SMALL
                #include "DepthOfField.cginc"
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_MEDIUM
                #include "DepthOfField.cginc"
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_LARGE
                #include "DepthOfField.cginc"
            ENDCG
        }

        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragBlur
                #define KERNEL_VERYLARGE
                #include "DepthOfField.cginc"
            ENDCG
        }

        // (6) Postfilter blur
        Pass
        {
            CGPROGRAM
                #pragma vertex VertDOF
                #pragma fragment FragPostBlur
                #include "DepthOfField.cginc"
            ENDCG
        }
    }

    FallBack Off
}
