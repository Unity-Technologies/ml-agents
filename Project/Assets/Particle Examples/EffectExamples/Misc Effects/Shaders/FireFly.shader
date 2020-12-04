Shader "Unlit/FireFlyNew"
{
	Properties
	{
		_MainTex ("Albedo (Emissive)", 2D) = "white" {}
		_Wings ("Wings", 2D) = "white" {}
		_EmissiveAmount ("_EmissiveAmount", Float) =2.0
	}
	SubShader
	{
		Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert alpha:fade
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float4 uv : TEXCOORD0;
				float4 color : COLOR;
			};

			struct v2f
			{
				float4 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float4 color : COLOR;
			};

			sampler2D _MainTex;
			float _EmissiveAmount;
			//float4 _MainTex_ST;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				//o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				o.uv = v.uv;
				o.color.rgba = v.color.rgba;
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 col = tex2D(_MainTex, i.uv);
				
				float4 emissive = i.color * col.a;
				
				clip(i.color.a - 0.5);

				col += emissive * pow(_EmissiveAmount, 2.2);
				return col;
			}
			ENDCG
			}

		Pass
			{
			Cull Off
			Blend SrcAlpha OneMinusSrcAlpha

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"
			
			struct appdata
			{
				float4 vertex : POSITION;
				float4 uv : TEXCOORD0;
				float4 color : COLOR;
			};

			struct v2f
			{
				float4 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float4 color : COLOR;
			};

			sampler2D _Wings;
			//float4 _MainTex_ST;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				//o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				o.uv = v.uv;
				o.color.rgba = v.color.rgba;
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 wings = tex2D(_Wings, i.uv.zw);				
				wings.a = lerp( wings.a, 0, i.color.a);

				return wings;
			}
			ENDCG
			
		}
	}
}
