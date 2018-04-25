Shader "Post Processing/Custom Motion Vector Texture"
{
    Properties
    {
        _MotionTex ("Motion Vector Texture", 2D) = "black" {}
        _MotionAmount ("Motion Vector Multiplier", range (-0.25, 0.25)) = 0
    }
    SubShader
    {
        Pass
        {
            Name "Motion Vectors"
            Tags { "LightMode" = "MotionVectors" }
 
            ZTest LEqual Cull Back ZWrite On
 
            CGPROGRAM

                #pragma vertex vert
                #pragma fragment FragMotionVectors
                #include "UnityCG.cginc"

                float4 _MotionValue;
                sampler2D _MotionTex;
                float4 _MotionTex_ST;
                float _MotionAmount;

                struct appdata
                {
                    float4 vertex : POSITION;
                    float2 uv : TEXCOORD0;
                    float3 normal : NORMAL;
                    float4 tangent : TANGENT;
                };

                struct v2f
                {
                    float2 uv : TEXCOORD0;
                    float4 vertex : SV_POSITION;
                    float3 normal : NORMAL;
                    float4 tangent : TANGENT;
                    float4 transposedTangent : TEXCOORD1;
                };

                v2f vert (appdata v)
                {
                    v2f o;
                    o.vertex = UnityObjectToClipPos(v.vertex);
                    o.uv = TRANSFORM_TEX(v.uv, _MotionTex);
                    o.normal = UnityObjectToClipPos(v.normal);
                    o.normal = o.normal * 0.5 + 0.5;
                    o.tangent = mul(UNITY_MATRIX_MV, v.tangent);
                    o.transposedTangent = (mul(UNITY_MATRIX_IT_MV, v.tangent)) * 0.5 + 0.5;
                    return o;
                }
 
                float4 FragMotionVectors(v2f i) : SV_Target
                {
                    half4 c = tex2D(_MotionTex, i.uv);
                    c.rg = (c.rg * 2.0 - 1.0) * _MotionAmount; // Using color texture so need to make 0.5 neutral
                    half4 t1 = i.tangent * 0.005; // Sides of tire
                    half4 t2 = c * float4(i.transposedTangent.r * 2.0, i.transposedTangent.g * 2.0, 0.0, 1.0); // Front of tire
                    half4 t3 = lerp(t2, t1, c.b); // Lerp between front and side of tire
                    return t3 * _MotionAmount;
                }
                
            ENDCG
        }
    }
}
