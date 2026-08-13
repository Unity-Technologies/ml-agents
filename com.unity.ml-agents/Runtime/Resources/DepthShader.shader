Shader "Custom/DepthShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 screenPos: TEXTCOORD1;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.screenPos = ComputeScreenPos(o.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex, _CameraDepthTexture;

            float4 frag (v2f i) : SV_Target
            {
                // Extract color from texture
                float4 color = tex2D(_MainTex, i.uv);

                // Extract depth from camera depth texture
                float depth = LinearEyeDepth(tex2D(_CameraDepthTexture, i.screenPos.xy));

                // Clip depth to far plane
                float farPlane = _ProjectionParams.z;
                if (depth > farPlane) depth = 0;

                // Convert color from linear to sRGB
                color.rgb = LinearToGammaSpace(saturate(color.rgb));

                // Store depth in alpha channel
                color.a = depth;

                return color;
            }
            ENDCG
        }
    }
}
