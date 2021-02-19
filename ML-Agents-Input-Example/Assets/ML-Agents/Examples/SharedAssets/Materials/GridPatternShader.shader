// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "ML-Agents/GridPattern" {
	Properties {
		_LineColor ("Line Color", Color) = (1,1,1,1)
		_CellColor ("Cell Color", Color) = (0,0,0,0)
		// _SelectedColor ("Selected Color", Color) = (1,0,0,1)
		[PerRendererData] _MainTex ("Albedo (RGB)", 2D) = "white" {}
		[IntRange] _GridSize("Grid Size", Range(1,100)) = 10
		_LineSize("Line Size", Range(0,1)) = 0.15
		// [FloatRange] _LineOffset("Line Offset", Range(0,1)) = 0
		[IntRange] _DrawU("Draw U Toggle ( 0 = False , 1 = True )", Range(0,1)) = 1
		[IntRange] _DrawV("Draw V Toggle ( 0 = False , 1 = True )", Range(0,1)) = 1
		// [IntRange] _SelectCell("Select Cell Toggle ( 0 = False , 1 = True )", Range(0,1)) = 0.0
		// [IntRange] _SelectedCellX("Selected Cell X", Range(0,100)) = 0.0
		// [IntRange] _SelectedCellY("Selected Cell Y", Range(0,100)) = 0.0
	}
	SubShader {
		Tags { "Queue"="AlphaTest" "RenderType"="TransparentCutout" }
		LOD 200
	

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0

		sampler2D _MainTex;

		struct Input {
			float2 uv_MainTex;
		};

		half _Glossiness = 0.0;
		half _Metallic = 0.0;
		float4 _LineColor;
		float4 _CellColor;
		// float4 _SelectedColor;

		float _GridSize;
		// float _LineOffset;
		float _LineSize;

		float _DrawU;
		float _DrawV;
		// float _SelectCell;
		// float _SelectedCellX;
		// float _SelectedCellY;

		// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
		// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
		// #pragma instancing_options assumeuniformscaling
		UNITY_INSTANCING_BUFFER_START(Props)
			// put more per-instance properties here
		UNITY_INSTANCING_BUFFER_END(Props)

		void surf (Input IN, inout SurfaceOutputStandard o) {
			// Albedo comes from a texture tinted by color

			float2 uv = IN.uv_MainTex;

			// _SelectedCellX = floor(_SelectedCellX);
			// _SelectedCellY = floor(_SelectedCellY);

			fixed4 c = float4(0.0,0.0,0.0,0.0);

			float brightness = 1.0;

			float gsize = floor(_GridSize);



			gsize += _LineSize;

			float2 id;

			id.x = floor(uv.x/(1.0/gsize));
			// id.y = floor(uv.y/(1.0/gsize));
			id.y = floor(uv.y/(1.0/gsize));

			float4 color = _CellColor;
			brightness = _CellColor.w;

			// //This checks that the cell is currently selected if the Select Cell slider is set to 1 ( True )
			// if (round(_SelectCell) == 1.0 && id.x == _SelectedCellX && id.y == _SelectedCellY)
			// {
			// 	brightness = _SelectedColor.w;
			// 	color = _SelectedColor;
			// }

			// if (frac(uv.x*gsize) <= _LineSize || frac(uv.y*gsize) <= _LineSize)
			// {
			// 	brightness = _LineColor.w;
			// 	color = _LineColor;
			// }
			if(round(_DrawU) == 1.0)
			{
				if (frac(uv.x*gsize) <= _LineSize)
				{
					// if (frac(uv.x*gsize) <= _LineSize)
					// {
					brightness = _LineColor.w;
					color = _LineColor;
				}
			}
			if(round(_DrawV) == 1.0)
			{
				if (frac(uv.y*gsize) <= _LineSize)
				{
					brightness = _LineColor.w;
					color = _LineColor;
				}
			}
			// // if (round(uv.x*gsize) == .8)
			// // if (uv.x >= _LineOffset)
			// if (uv.x >= .2 && uv.x <= .3)
			// {
			// 	brightness = _LineColor.w;
			// 	color = _LineColor;
			// 	// color = _CellColor;
			// 	// brightness = _CellColor.w;
			// }

			// if (frac(uv.x*gsize) <= _LineSize && frac(uv.y*gsize) <= _LineSize)
			// {
			// 	// brightness = _LineColor.w;
			// 	// color = _LineColor;
			// 	color = _CellColor;
			// 	brightness = _CellColor.w;
			// }


			// if (frac(uv.x*gsize/_LineOffset) <= _LineSize)
			// {
			// 	brightness = _LineColor.w;
			// 	color = _LineColor;
			// }
			// if (frac(uv.x*gsize/.5) <= _LineSize || fradc(uv.y*gsize) <= _LineSize)
			// {
			// 	brightness = _LineColor.w;
			// 	color = _LineColor;
			// }
			

			//Clip transparent spots using alpha cutout
			if (brightness == 0.0) {
				clip(c.a - 1.0);
			}
			

			o.Albedo = float4( color.x*brightness,color.y*brightness,color.z*brightness,brightness);
			// Metallic and smoothness come from slider variables
			o.Metallic = 0.0;
			o.Smoothness = 0.0;
			o.Alpha = 0.0;
		}
		ENDCG
	}
	FallBack "Diffuse"
}
