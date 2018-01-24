//======= Copyright (c) Valve Corporation, All rights reserved. ===============
// UNITY_SHADER_NO_UPGRADE
Shader "Custom/SteamVR_ClearAll" {
	Properties
	{
		_Color ("Color", Color) = (0, 0, 0, 0)
		_MainTex ("Base (RGB)", 2D) = "white" {}
	}

	CGINCLUDE

	#include "UnityCG.cginc"

	float4 _Color;
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

	float4 frag(v2f i) : COLOR {
		return _Color;
	}

	ENDCG

	SubShader {
		Tags{ "Queue" = "Background" }
		Pass {
			ZTest Always Cull Off ZWrite On
			Fog { Mode Off }

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			ENDCG
		}
	}
}

