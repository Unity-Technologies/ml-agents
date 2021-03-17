Shader "Hidden/Post FX/Blit"
{
    Properties
    {
        _MainTex("Main Texture", 2D) = "white" {}
    }

    CGINCLUDE

        #include "UnityCG.cginc"
        #include "Common.cginc"

        struct Varyings
        {
            float2 uv : TEXCOORD0;
            float4 vertex : SV_POSITION;
        };

        Varyings VertBlit(AttributesDefault v)
        {
            Varyings o;
            o.vertex = UnityObjectToClipPos(v.vertex);
            o.uv = UnityStereoScreenSpaceUVAdjust(v.texcoord, _MainTex_ST);
            return o;
        }

        half4 FragBlit(Varyings i) : SV_Target
        {
            half4 col = tex2D(_MainTex, i.uv);
            return col;
        }

    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM

                #pragma vertex VertBlit
                #pragma fragment FragBlit

            ENDCG
        }
    }
}
