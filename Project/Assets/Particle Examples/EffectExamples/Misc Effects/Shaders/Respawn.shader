// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "Custom/Respawn" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		[HDR]_Emission ("Emission", Color) = (0,0,0,0)
		_MainTex ("Albedo", 2D) = "white" {}
		_Normal ("Normal", 2D) = "bump" {}
		_MetallicSmooth ("Metallic (RGB) Smooth (A)", 2D) = "white" {}
		_AO ("AO", 2D) = "white" {}
		[HDR]_EdgeColor1 ("Edge Color", Color) = (1,1,1,1)
		_Noise ("Noise", 2D) = "white" {}
		[Toggle] _Use_Gradient ("Use Gradient?", Float) = 1
		_Gradient ("Gradient", 2D) = "white" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0
		[PerRendererData]_Cutoff ("Cutoff", Range(0,1)) = 0.0
		_EdgeSize ("EdgeSize", Range(0,1)) = 0.2
		_NoiseStrength ("Noise Strength", Range(0,1)) = 0.4
		_DisplaceAmount ("Displace Amount", Float) = 1.5
		_cutoff ("cutoff", Range(0,1)) = 0.0

		
	}
	SubShader {
		Tags { "Queue"="AlphaTest" "RenderType"="TransparentCutout" "IgnoreProjector"="True" }
		Cull Off

		LOD 200
		
		CGPROGRAM

		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows vertex:vert addshadow 

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		#pragma multi_compile __ _USE_GRADIENT_ON

		sampler2D _MainTex;
		sampler2D _Noise;
		sampler2D _Gradient;
		sampler2D _Normal;
		sampler2D _MetallicSmooth;
		sampler2D _AO;

		struct Input {
			float2 uv_Noise;
			float2 uv_MainTex;
			fixed4 color : COLOR0;
			float3 worldPos;
		};


		half _Glossiness, _Metallic, _Cutoff, _EdgeSize, _NoiseStrength, _DisplaceAmount;
		half _cutoff;
		half4 _Color, _EdgeColor1, _Emission;


		void vert (inout appdata_full v, out Input o) {
        	UNITY_INITIALIZE_OUTPUT(Input,o);
        	float3 pos = mul((float3x3)unity_ObjectToWorld, v.vertex.xyz);
        	//pos.x += _cutoff*5;
        	float4 tex = tex2Dlod(_Noise, float4(pos, 0)*0.5);
    


        	float4 Gradient = tex2Dlod (_Gradient, float4(v.texcoord.xy,0,0));
        	float mask = smoothstep(_Cutoff, _Cutoff - 0.3, 1-Gradient);

  
        	float displacementAmount = lerp(0, _DisplaceAmount, _cutoff);
        	//v.vertex.xyz += mul((float3x3)unity_WorldToObject, float3(0, vertexOffset, 0));
        	//v.vertex.xyz = lerp(float3(0,0,0), v.vertex.xyz, _cutoff);
    	  }

		// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
		// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
		// #pragma instancing_options assumeuniformscaling
		UNITY_INSTANCING_BUFFER_START(Props)
			// put more per-instance properties here
		UNITY_INSTANCING_BUFFER_END(Props)

		void surf (Input IN, inout SurfaceOutputStandard o) {
			float2 uv = IN.uv_Noise;
			uv.y += _cutoff;
			half3 Noise = tex2D (_Noise, uv);
			Noise.r = lerp(0, 1, Noise.r);
			half4 MetallicSmooth = tex2D (_MetallicSmooth, IN.uv_MainTex);
			//_cutoff  = lerp(0, _cutoff, _cutoff);
			_cutoff = 1-_cutoff;

			#if _USE_GRADIENT_ON
			half3 Gradient = tex2D (_Gradient, IN.uv_MainTex);
			half Edge = smoothstep(_Cutoff, _Cutoff - _EdgeSize, 1-(Gradient + Noise.r*(1-Gradient)*_NoiseStrength));
			#else
			half Edge = smoothstep(_cutoff + _EdgeSize, _cutoff, clamp(Noise.r, _EdgeSize, 1));
			#endif

			fixed4 c = tex2D (_MainTex, IN.uv_MainTex);
			fixed3 EmissiveCol = c.a * _Emission;

			o.Albedo = _Color;
			o.Occlusion = tex2D (_AO, IN.uv_MainTex);
			o.Emission = EmissiveCol + _EdgeColor1 * Edge;
			o.Normal = UnpackNormal (tex2D (_Normal, IN.uv_MainTex));
			o.Metallic = MetallicSmooth.r * _Metallic;
			o.Smoothness = MetallicSmooth.a * _Glossiness;
			clip(Noise - _cutoff);
		}
		ENDCG
	}
	FallBack "Diffuse"
}
