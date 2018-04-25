Shader "Hidden/Post FX/Fog"
{
    Properties
    {
        _MainTex("Main Texture", 2D) = "white" {}
    }

    CGINCLUDE

        #pragma multi_compile __ FOG_LINEAR FOG_EXP FOG_EXP2
        #include "UnityCG.cginc"
        #include "Common.cginc"

        #define SKYBOX_THREASHOLD_VALUE 0.9999

        struct Varyings
        {
            float2 uv : TEXCOORD0;
            float4 vertex : SV_POSITION;
        };

        Varyings VertFog(AttributesDefault v)
        {
            Varyings o;
            o.vertex = UnityObjectToClipPos(v.vertex);
            o.uv = UnityStereoScreenSpaceUVAdjust(v.texcoord, _MainTex_ST);
            return o;
        }

        sampler2D _CameraDepthTexture;

        half4 _FogColor;
        float _Density;
        float _Start;
        float _End;

        half ComputeFog(float z)
        {
            half fog = 0.0;
        #if FOG_LINEAR
            fog = (_End - z) / (_End - _Start);
        #elif FOG_EXP
            fog = exp2(-_Density * z);
        #else // FOG_EXP2
            fog = _Density * z;
            fog = exp2(-fog * fog);
        #endif
            return saturate(fog);
        }

        float ComputeDistance(float depth)
        {
            float dist = depth * _ProjectionParams.z;
            dist -= _ProjectionParams.y;
            return dist;
        }

        half4 FragFog(Varyings i) : SV_Target
        {
            half4 color = tex2D(_MainTex, i.uv);

            float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
            depth = Linear01Depth(depth);
            float dist = ComputeDistance(depth);
            half fog = 1.0 - ComputeFog(dist);

            return lerp(color, _FogColor, fog);
        }

        half4 FragFogExcludeSkybox(Varyings i) : SV_Target
        {
            half4 color = tex2D(_MainTex, i.uv);

            float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
            depth = Linear01Depth(depth);
            float skybox = depth < SKYBOX_THREASHOLD_VALUE;
            float dist = ComputeDistance(depth);
            half fog = 1.0 - ComputeFog(dist);

            return lerp(color, _FogColor, fog * skybox);
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM

                #pragma vertex VertFog
                #pragma fragment FragFog

            ENDCG
        }

        Pass
        {
            CGPROGRAM

                #pragma vertex VertFog
                #pragma fragment FragFogExcludeSkybox

            ENDCG
        }
    }
}
