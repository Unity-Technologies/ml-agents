Shader "Custom/Ice" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_FresnelColor ("_FresnelColor", Color) = (1,1,1)
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0
		_BumpAmt  ("Distortion", range (0,1000)) = 10
		_BumpMap ("Normalmap", 2D) = "bump" {}
		_Fresnel ("Fresnel", Range(1.0,12.0)) = 3.0
		_marchDistance ("March Distance", Float) = 3.0
		_numSteps ("Steps", Float) = 4.0
		_Ramp ("Ramp", 2D) = "white" {}
		_InnerRamp ("_InnerRamp", 2D) = "white" {}
	}
	SubShader {
		Tags { "Queue"="Transparent" "RenderType"="Transparent" }
		LOD 200



		GrabPass {
			Name "BASE"
			Tags { "LightMode" = "Always" }
		}

		Pass {
			Name "BASE"

			
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fog
			#include "UnityCG.cginc"


			struct appdata_t {
				float4 vertex : POSITION;
				float2 texcoord: TEXCOORD0;
			};

			struct v2f {
				float4 vertex : SV_POSITION;
				float4 uvgrab : TEXCOORD0;
				float2 uvbump : TEXCOORD1;
				float2 uvmain : TEXCOORD2;
				UNITY_FOG_COORDS(3)
			};

			float _BumpAmt;

			float4 _BumpMap_ST;
			float4 _MainTex_ST;

			v2f vert (appdata_t v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				#if UNITY_UV_STARTS_AT_TOP
				float scale = -1.0;
				#else
				float scale = 1.0;
				#endif
				o.uvgrab.xy = (float2(o.vertex.x, o.vertex.y*scale) + o.vertex.w) * 0.5;
				o.uvgrab.zw = o.vertex.zw;
				o.uvbump = TRANSFORM_TEX( v.texcoord, _BumpMap );
				o.uvmain = TRANSFORM_TEX( v.texcoord, _MainTex );
				UNITY_TRANSFER_FOG(o,o.vertex);
				return o;
			}

			sampler2D _GrabTexture;
			float4 _GrabTexture_TexelSize;
			sampler2D _BumpMap;
			sampler2D _MainTex;

			half4 frag (v2f i) : SV_Target
			{
				// calculate perturbed coordinates
				half2 bump = UnpackNormal(tex2D( _BumpMap, i.uvbump )).rg; // we could optimize this by just reading the x & y without reconstructing the Z
				float2 offset = bump * _BumpAmt * _GrabTexture_TexelSize.xy;
				i.uvgrab.xy = offset + i.uvgrab.xy;
				
				half4 col = tex2Dproj( _GrabTexture, UNITY_PROJ_COORD(i.uvgrab));
				//half4 tint = tex2D(_MainTex, i.uvmain);
				//col *= tint;
				UNITY_APPLY_FOG(i.fogCoord, col);
				return col;
			}
			ENDCG
		}

		//Blend SrcBlend OneMinusSrcAlpha
		CGPROGRAM

		// Physically based Standard lighting model, and enable shadows on all light types
		//#pragma surface surf Standard alpha:fade fullforwardshadows
		#pragma surface surf RampSpec alpha:fade fullforwardshadows


		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		#include "UnityPBSLighting.cginc"

		sampler2D _MainTex;
		sampler2D _BumpMap;
		sampler2D _Ramp, _InnerRamp;

		struct Input {
			float2 uv_MainTex;
			float2 uv_BumpMap;
			float3 viewDir;
		};

		half _Glossiness;
		half _Metallic;
		fixed4 _Color;
		float _Fresnel;
		fixed3 _FresnelColor;
		fixed _marchDistance, _numSteps;


		half3 RampShading (float3 normal, half3 lightDir, half3 viewDir, half3 lightCol) {
	        half NdotL = dot (normal, lightDir);
	        half diff = NdotL * 0.5 + 0.5;
	        half3 ramp = tex2D (_Ramp, float2(diff, 0.5)).rgb;
	        half3 c;
	        c.rgb = ramp * lightCol;
	        return c;

	    }

	    half4 LightingRampSpec(SurfaceOutputStandardSpecular s, half3 viewDir, UnityGI gi)
		{
			s.Normal = normalize(s.Normal);

			// energy conservation
			half oneMinusReflectivity;
			s.Albedo = EnergyConservationBetweenDiffuseAndSpecular(s.Albedo, s.Specular, /*out*/ oneMinusReflectivity);

			// shader relies on pre-multiply alpha-blend (_SrcBlend = One, _DstBlend = OneMinusSrcAlpha)
			// this is necessary to handle transparency in physically correct way - only diffuse component gets affected by alpha
			half outputAlpha;
			s.Albedo = PreMultiplyAlpha(s.Albedo, s.Alpha, oneMinusReflectivity, /*out*/ outputAlpha);

			half4 c = UNITY_BRDF_PBS(s.Albedo, s.Specular, oneMinusReflectivity, s.Smoothness, s.Normal, viewDir, gi.light, gi.indirect);

			c.rgb += RampShading(s.Normal, gi.light.dir, viewDir,  gi.light.color) * _Color;
			//c.rgb += SubsurfaceShadingSimple(_InternalColor, s.Normal, viewDir, s.Alpha*_SSS, gi.light.dir, gi.light.color);

			c.a = outputAlpha;
			return c;
		}

		inline void LightingRampSpec_GI(
			SurfaceOutputStandardSpecular s,
			UnityGIInput data,
			inout UnityGI gi)
		{
		#if defined(UNITY_PASS_DEFERRED) && UNITY_ENABLE_REFLECTION_BUFFERS
				gi = UnityGlobalIllumination(data, s.Occlusion, s.Normal);
		#else
				Unity_GlossyEnvironmentData g = UnityGlossyEnvironmentSetup(s.Smoothness, data.worldViewDir, s.Normal, s.Specular);
				gi = UnityGlobalIllumination(data, s.Occlusion, s.Normal, g);
		#endif
		}

		void surf(Input IN, inout SurfaceOutputStandardSpecular o) {

			// Albedo comes from a texture tinted by color
			fixed4 c = _Color;
			//half2 bump = UnpackNormal(tex2D( _BumpMap, i.uvbump )).rg;

			//Inner structure parallax
			float3 InnerStructure = float3(0, 0, 0);
			float2 UV = IN.uv_MainTex;
			float offset =  1;
			for (float d = 0.0; d < _marchDistance; d += _marchDistance / _numSteps)
			{
				UV -= (IN.viewDir*d)/_numSteps *  tex2D (_MainTex, IN.uv_MainTex).g;
				float4 Ldensity = tex2D(_MainTex, UV).r;
				InnerStructure += saturate(Ldensity[0])*tex2D(_InnerRamp, float2(1/_numSteps * offset, 0.5));
				offset ++;
			}


			// Metallic and smoothness come from slider variables
			o.Normal = UnpackScaleNormal (tex2D (_BumpMap, IN.uv_BumpMap), 0.2);

			half rim = saturate(dot (normalize(IN.viewDir), o.Normal));
			half rim2 = 1 - saturate(dot (normalize(IN.viewDir), o.Normal));
			float fresnel = pow(rim2, _Fresnel);// + pow(rim2, _Fresnel);

			o.Alpha = clamp(c.a + fresnel + InnerStructure, 0, 1) + 0.2;
			o.Albedo = _Color + InnerStructure * _FresnelColor;


			o.Specular = _Metallic;
			o.Smoothness = _Glossiness;
			//o.Emission = SubsurfaceShadingSimple(_InternalColor, o.Normal, IN.viewDir, c.a*_SSS, IN.lightDir, _LightColor0);
		}

		ENDCG
		
		
	}

	FallBack "Diffuse"
}
