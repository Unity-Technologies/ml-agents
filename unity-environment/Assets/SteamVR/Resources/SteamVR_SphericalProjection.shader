//======= Copyright (c) Valve Corporation, All rights reserved. ===============
// UNITY_SHADER_NO_UPGRADE
Shader "Custom/SteamVR_SphericalProjection" {
	Properties {
		_MainTex ("Base (RGB)", 2D) = "white" {}
		_N ("N (normal of plane)", Vector) = (0,0,0,0)
		_Phi0 ("Phi0", Float) = 0
		_Phi1 ("Phi1", Float) = 1
		_Theta0 ("Theta0", Float) = 0
		_Theta1 ("Theta1", Float) = 1
		_UAxis ("uAxis", Vector) = (0,0,0,0)
		_VAxis ("vAxis", Vector) = (0,0,0,0)
		_UOrigin ("uOrigin", Vector) = (0,0,0,0)
		_VOrigin ("vOrigin", Vector) = (0,0,0,0)
		_UScale ("uScale", Float) = 1
		_VScale ("vScale", Float) = 1
	}

	CGINCLUDE

	#include "UnityCG.cginc"

	sampler2D _MainTex;
	float4 _N;
	float _Phi0, _Phi1, _Theta0, _Theta1;
	float4 _UAxis, _VAxis;
	float4 _UOrigin, _VOrigin;
	float _UScale, _VScale;

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
		o.tex = float2(
			lerp(_Phi0, _Phi1, v.texcoord.x),
			lerp(_Theta0, _Theta1, v.texcoord.y));
		return o;
	}

	float3 cartesian(float phi, float theta)
	{
		float sinTheta = sin(theta);
		return float3(
			sinTheta * sin(phi),
			cos(theta),
			sinTheta * cos(phi));
	}

	float4 frag(v2f i) : COLOR {
		float3 V = cartesian(i.tex.x, i.tex.y);
		float3 P = V / dot(V, _N.xyz); // intersection point on plane
		float2 uv = float2(
			dot(P - _UOrigin.xyz, _UAxis.xyz) * _UScale,
			dot(P - _VOrigin.xyz, _VAxis.xyz) * _VScale);
		return tex2D(_MainTex, uv);
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
