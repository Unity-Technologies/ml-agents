Shader "Custom/WireFrame" {
	Properties {
		_LineColor ("LineColor", Color) = (1, 1, 1, 1)
		_MainColor ("_MainColor", Color) = (1, 1, 1, 1)
		_LineWidth ("Line width", Range(0, 1)) = 0.1
		_ParcelSize ("ParcelSize", Range(0, 100)) = 1
	}
	SubShader {
		Tags { "Queue"="Transparent" "RenderType"="Transparent" }
		
		CGPROGRAM
		#pragma surface surf Lambert alpha

		sampler2D _MainTex;
		float4 _LineColor;
		float4 _MainColor;
		fixed _LineWidth;
		float _ParcelSize;

		struct Input {
			float2 uv_MainTex;
			float3 worldPos;
		};

		void surf (Input IN, inout SurfaceOutput o) {
			half val1 = step(_LineWidth * 2, frac(IN.worldPos.x / _ParcelSize) + _LineWidth);
			half val2 = step(_LineWidth * 2, frac(IN.worldPos.z / _ParcelSize) + _LineWidth);
			fixed val = 1 - (val1 * val2);
			o.Albedo = lerp(_MainColor, _LineColor, val);
			o.Alpha = 1;
		}
		ENDCG
	} 
	FallBack "Diffuse"
}

// //Shader uses screen-space partial derivatives, works the best with terrain meshes.

// Shader "Wireframe"
// {
// 	Properties
// 	{
// 		[Header(Settings)] [Toggle] _transparency ("Transparency", Float) = 1					
// 	}
// 	Subshader
// 	{
// 		Pass
// 		{
// 			Cull Off
// 			CGPROGRAM
// 			#pragma vertex vertex_shader
// 			#pragma fragment pixel_shader
// 			#pragma target 3.0
			
// 			struct structure
// 			{
// 				float4 gl_Position : SV_POSITION;
// 				float3 vertex : TEXCOORD0;
// 			};

// 			float _transparency;
			
// 			structure vertex_shader (float4 vertex:POSITION) 
// 			{
// 				structure vs;
// 				vs.gl_Position = UnityObjectToClipPos (vertex);
// 				vs.vertex = vertex;
// 				return vs;
// 			}

// 			float4 pixel_shader (structure ps) : COLOR
// 			{
// 				float2 p = ps.vertex.xz;
// 				float2 g = abs(frac(p - 0.5) - 0.5) / fwidth(p);
// 				float s = min(g.x, g.y);
// 				float4 c =  float4(s,s,s, 1.0);	
// 				if (c.r<1.0)
// 					return 1.0-c;
// 				else
// 				{
// 					if (_transparency==1) discard;
// 					return 0;
// 				}
// 			}
// 			ENDCG
// 		}
// 	}
// }