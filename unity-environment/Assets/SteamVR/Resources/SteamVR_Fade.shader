//======= Copyright (c) Valve Corporation, All rights reserved. ===============
// UNITY_SHADER_NO_UPGRADE
Shader "Custom/SteamVR_Fade"
{
	SubShader
	{
		Pass
		{
			Blend SrcAlpha OneMinusSrcAlpha
			ZTest Always
			Cull Off
			ZWrite Off

			CGPROGRAM
				#pragma vertex MainVS
				#pragma fragment MainPS

				float4 fadeColor;

				float4 MainVS( float4 vertex : POSITION ) : SV_POSITION
				{
					return vertex.xyzw;
				}

				float4 MainPS() : SV_Target
				{
					return fadeColor.rgba;
				}
			ENDCG
		}
	}
}