//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: 
//
//=============================================================================
// UNITY_SHADER_NO_UPGRADE
Shader "Valve/VR/ControllerButtonHints"
{
	Properties
	{
		_MainTex ( "Texture", 2D ) = "white" {}
		_Color( "Color", Color ) = ( 1, 1, 1, 1 )
		_SceneTint( "SceneTint", Color ) = ( 1, 1, 1, 1 )
	}
	SubShader
	{
		Tags{ "Queue" = "Transparent+1" "RenderType" = "Transparent" }
		LOD 100
		Pass
		{
			// Render State ---------------------------------------------------------------------------------------------------------------------------------------------
			Blend Zero SrcColor // Alpha blending
			Cull Off
			ZWrite Off
			ZTest Off
			Stencil
			{
				Ref 2
				Comp notequal
				Pass replace
				Fail keep
			}

			CGPROGRAM

			#pragma vertex MainVS
			#pragma fragment MainPS
			
			// Includes -------------------------------------------------------------------------------------------------------------------------------------------------
			#include "UnityCG.cginc"

			// Structs --------------------------------------------------------------------------------------------------------------------------------------------------
			struct VertexInput
			{
				float4 vertex : POSITION;
			};

			struct VertexOutput
			{
				float4 vertex : SV_POSITION;
			};

			// Globals --------------------------------------------------------------------------------------------------------------------------------------------------
			float4 _SceneTint;
			
			// MainVs ---------------------------------------------------------------------------------------------------------------------------------------------------
			VertexOutput MainVS( VertexInput i )
			{
				VertexOutput o;
#if UNITY_VERSION >= 540
				o.vertex = UnityObjectToClipPos(i.vertex);
#else
				o.vertex = mul(UNITY_MATRIX_MVP, i.vertex);
#endif				
				return o;
			}
			
			// MainPs ---------------------------------------------------------------------------------------------------------------------------------------------------
			float4 MainPS( VertexOutput i ) : SV_Target
			{
				return _SceneTint.rgba;
			}

			ENDCG
		}
		Pass
		{
			// Render State ---------------------------------------------------------------------------------------------------------------------------------------------
			Blend SrcAlpha OneMinusSrcAlpha // Alpha blending
			Cull Off
			ZWrite Off
			ZTest Always

			CGPROGRAM

			#pragma vertex MainVS
			#pragma fragment MainPS
			
			// Includes -------------------------------------------------------------------------------------------------------------------------------------------------
			#include "UnityCG.cginc"

			// Structs --------------------------------------------------------------------------------------------------------------------------------------------------
			struct VertexInput
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct VertexOutput
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			// Globals --------------------------------------------------------------------------------------------------------------------------------------------------
			sampler2D _MainTex;
			float4 _MainTex_ST;
			float4 _Color;
			
			// MainVs ---------------------------------------------------------------------------------------------------------------------------------------------------
			VertexOutput MainVS( VertexInput i )
			{
				VertexOutput o;
#if UNITY_VERSION >= 540
				o.vertex = UnityObjectToClipPos(i.vertex);
#else
				o.vertex = mul(UNITY_MATRIX_MVP, i.vertex);
#endif
				o.uv = TRANSFORM_TEX( i.uv, _MainTex );
				
				return o;
			}
			
			// MainPs ---------------------------------------------------------------------------------------------------------------------------------------------------
			float4 MainPS( VertexOutput i ) : SV_Target
			{
				float4 vColor;
				vColor.rgb = lerp( tex2D(_MainTex, i.uv).rgb, _Color.rgb, _Color.a );
				vColor.a = _Color.a;

				return vColor.rgba;
			}

			ENDCG
		}
	}
}
