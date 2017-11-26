//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Used for objects that can be seen through objects in front of them
//
//=============================================================================
// UNITY_SHADER_NO_UPGRADE
Shader "Valve/VR/SeeThru"
{
	Properties
	{
		_Color( "Color", Color ) = ( 1, 1, 1, 1 )
	}
	SubShader
	{
		Tags{ "Queue" = "Geometry+1" "RenderType" = "Transparent" }
		LOD 100

		Pass
		{
			// Render State ---------------------------------------------------------------------------------------------------------------------------------------------
			Blend SrcAlpha OneMinusSrcAlpha // Alpha blending
			Cull Off
			ZWrite Off
			ZTest Greater
			Stencil
			{
				Ref 2
				Comp notequal
				Pass replace
				Fail keep
			}

			CGPROGRAM
				#pragma target 5.0
				#pragma only_renderers d3d11 vulkan
				#pragma exclude_renderers gles

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
					float4 vColor = _Color.rgba;
				
					return vColor.rgba;
				}

			ENDCG
		}
	}
}
