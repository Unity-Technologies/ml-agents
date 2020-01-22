// Simplified SDF shader:
// - No Shading Option (bevel / bump / env map)
// - No Glow Option
// - Softness is applied on both side of the outline

Shader "Hidden/TextMeshPro/Internal/Distance Field SSD" {

Properties {
	_FaceColor			("Face Color", Color) = (1,1,1,1)
	_FaceDilate			("Face Dilate", Range(-1,1)) = 0

	_OutlineSoftness	("Outline Softness", Range(0,1)) = 0.02

	_WeightNormal		("Weight Normal", float) = 0
	_WeightBold			("Weight Bold", float) = .5

	_MainTex			("Font Atlas", 2D) = "white" {}
	_TextureWidth		("Texture Width", float) = 512
	_TextureHeight		("Texture Height", float) = 512
	_GradientScale		("Gradient Scale", float) = 5
	_ScaleX				("Scale X", float) = 1
	_ScaleY				("Scale Y", float) = 1
	_Sharpness			("Sharpness", Range(-1,1)) = 0

	_VertexOffsetX		("Vertex OffsetX", float) = 0
	_VertexOffsetY		("Vertex OffsetY", float) = 0
	
	_ColorMask			("Color Mask", Float) = 15
}

SubShader {
	Tags 
	{
		"ForceSupported" = "True"
	}

	Lighting Off
	Blend One OneMinusSrcAlpha
	Cull Off
	ZWrite Off
	ZTest Always

	Pass {
		CGPROGRAM
		#pragma vertex VertShader
		#pragma fragment PixShader

		#include "UnityCG.cginc"
		#include "TMP_Properties.cginc"

		sampler2D _GUIClipTexture;
		uniform float4x4 unity_GUIClipTextureMatrix;

		struct vertex_t {
			float4	vertex			: POSITION;
			float3	normal			: NORMAL;
			fixed4	color			: COLOR;
			float2	texcoord0		: TEXCOORD0;
			float2	texcoord1		: TEXCOORD1;
		};

		struct pixel_t {
			float4	vertex			: SV_POSITION;
			fixed4	faceColor		: COLOR;
			float2	texcoord0		: TEXCOORD0;
			float2	clipUV			: TEXCOORD1;
		};


		pixel_t VertShader(vertex_t input)
		{
			// Does not handle simulated bold correctly.
			
			float4 vert = input.vertex;
			vert.x += _VertexOffsetX;
			vert.y += _VertexOffsetY;
			float4 vPosition = UnityObjectToClipPos(vert);

			float opacity = input.color.a;

			fixed4 faceColor = fixed4(input.color.rgb, opacity) * _FaceColor;
			faceColor.rgb *= faceColor.a;

			// Generate UV for the Clip Texture
			float3 eyePos = UnityObjectToViewPos(input.vertex);
			float2 clipUV = mul(unity_GUIClipTextureMatrix, float4(eyePos.xy, 0, 1.0));

			// Structure for pixel shader
			pixel_t output = {
				vPosition,
				faceColor,
				float2(input.texcoord0.x, input.texcoord0.y),
				clipUV,
			};

			return output;
		}

		half transition(half2 range, half distance)
        {
            return smoothstep(range.x, range.y, distance);
        }

		// PIXEL SHADER
		fixed4 PixShader(pixel_t input) : SV_Target
		{
			half distanceSample = tex2D(_MainTex, input.texcoord0).a;
            half smoothing = fwidth(distanceSample) * (1 - _Sharpness) + _OutlineSoftness;
            half contour = 0.5 - _FaceDilate * 0.5;
            half2 edgeRange = half2(contour - smoothing, contour + smoothing);

			half4 c = input.faceColor;
            
            half edgeTransition = transition(edgeRange, distanceSample);
            c *= edgeTransition;

			c *= tex2D(_GUIClipTexture, input.clipUV).a;

			return c;
		}
		ENDCG
	}
}

CustomEditor "TMPro.EditorUtilities.TMP_SDFShaderGUI"
}
