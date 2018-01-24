//======= Copyright (c) Valve Corporation, All rights reserved. ===============
// UNITY_SHADER_NO_UPGRADE
Shader "Custom/SteamVR_ColorOut" {
	Properties { _MainTex ("Base (RGB)", 2D) = "white" {} }

	CGINCLUDE

	#include "UnityCG.cginc"

	sampler2D _MainTex;

	struct v2f {
		float4 pos : SV_POSITION;
		float2 tex : TEXCOORD0;
	};

	v2f vert(appdata_base v) {
		v2f o;
#if UNITY_VERSION >= 540
		o.pos = UnityObjectToClipPos(v.vertex);
#else
		o.pos = mul(UNITY_MATRIX_MVP, v.vertex);
#endif
		o.tex = v.texcoord;
		return o;
	}

	float luminance(float3 color)
	{
		return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
	}

	float4 frag(v2f i) : COLOR {
		float4 color = tex2D(_MainTex, i.tex);
		return float4(color.rgb, 1);
	}

	ENDCG

	SubShader {
		Pass {
			ZTest Always Cull Off ZWrite Off
			Fog { Mode Off }

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			ENDCG
		}
	}
}

