float2 UnpackUV(float uv)
{
	float2 output;
	output.x = floor(uv / 4096.0);
	output.y = uv - 4096.0 * output.x;

	return output * 0.001953125;
}

float4 BlendARGB(float4 overlying, float4 underlying)
{
	overlying.rgb *= overlying.a;
	underlying.rgb *= underlying.a;
	float3 blended = overlying.rgb + ((1 - overlying.a) * underlying.rgb);
	float alpha = underlying.a + (1 - underlying.a) * overlying.a;
	return float4(blended / alpha, alpha);
}

float3 GetSpecular(float3 n, float3 l)
{
	float spec = pow(max(0.0, dot(n, l)), _Reflectivity);
	return _SpecularColor.rgb * spec * _SpecularPower;
}

void GetSurfaceNormal_float(texture2D atlas, float textureWidth, float textureHeight, float2 uv, bool isFront, out float3 nornmal)
{
	float3 delta = float3(1.0 / textureWidth, 1.0 / textureHeight, 0.0);

	// Read "height field"
	float4 h = float4(
		SAMPLE_TEXTURE2D(atlas, SamplerState_Linear_Clamp, uv - delta.xz).a,
		SAMPLE_TEXTURE2D(atlas, SamplerState_Linear_Clamp, uv + delta.xz).a,
		SAMPLE_TEXTURE2D(atlas, SamplerState_Linear_Clamp, uv - delta.zy).a,
		SAMPLE_TEXTURE2D(atlas, SamplerState_Linear_Clamp, uv + delta.zy).a);

	bool raisedBevel = _BevelType;

	h += _BevelOffset;

	float bevelWidth = max(.01, _BevelWidth);

	// Track outline
	h -= .5;
	h /= bevelWidth;
	h = saturate(h + .5);

	if (raisedBevel) h = 1 - abs(h * 2.0 - 1.0);
	h = lerp(h, sin(h * 3.141592 / 2.0), float4(_BevelRoundness, _BevelRoundness, _BevelRoundness, _BevelRoundness));
	h = min(h, 1.0 - float4(_BevelClamp, _BevelClamp, _BevelClamp, _BevelClamp));
	h *= _BevelAmount * bevelWidth * _GradientScale * -2.0;

	float3 va = normalize(float3(-1.0, 0.0, h.y - h.x));
	float3 vb = normalize(float3(0.0, 1.0, h.w - h.z));

	float3 f = float3(1, 1, 1);
	if (isFront) f = float3(1, 1, -1);
	nornmal = cross(va, vb) * f;
}

void EvaluateLight_float(float4 faceColor, float3 n, out float4 color)
{
	n.z = abs(n.z);
	float3 light = normalize(float3(sin(_LightAngle), cos(_LightAngle), 1.0));

	float3 col = max(faceColor.rgb, 0) + GetSpecular(n, light)* faceColor.a;
	//faceColor.rgb += col * faceColor.a;
	col *= 1 - (dot(n, light) * _Diffuse);
	col *= lerp(_Ambient, 1, n.z * n.z);

	//fixed4 reflcol = texCUBE(_Cube, reflect(input.viewDir, -n));
	//faceColor.rgb += reflcol.rgb * lerp(_ReflectFaceColor.rgb, _ReflectOutlineColor.rgb, saturate(sd + outline * 0.5)) * faceColor.a;

	color = float4(col, faceColor.a);
}

// Add custom function to handle time in HDRP


//
void GenerateUV_float(float2 inUV, float4 transform, float2 animSpeed, out float2 outUV)
{
	outUV = inUV * transform.xy + transform.zw + (animSpeed * _Time.y);
}

void ComputeUVOffset_float(float texWidth, float texHeight, float2 offset, float SDR, out float2 uvOffset)
{
	uvOffset = float2(-offset.x * SDR / texWidth, -offset.y * SDR / texHeight);
}

void ScreenSpaceRatio2_float(float4x4 projection, float4 position, float2 objectScale, float screenWidth, float screenHeight, float fontScale, out float SSR)
{
	float2 pixelSize = position.w;
	pixelSize /= (objectScale * mul((float2x2)projection, float2(screenWidth, screenHeight)));
	SSR = rsqrt(dot(pixelSize, pixelSize)*2) * fontScale;
}

// UV			: Texture coordinate of the source distance field texture
// TextureSize	: Size of the source distance field texture
// Filter		: Enable perspective filter (soften)
void ScreenSpaceRatio_float(float2 UV, float TextureSize, bool Filter, out float SSR)
{
	if(Filter)
	{
		float2 a = float2(ddx(UV.x), ddy(UV.x));
		float2 b = float2(ddx(UV.y), ddy(UV.y));
		float s = lerp(dot(a,a), dot(b,b), 0.5);
		SSR = rsqrt(s) / TextureSize;
	}
	else
	{
		float s = rsqrt(abs(ddx(UV.x) * ddy(UV.y) - ddy(UV.x) * ddx(UV.y)));
		SSR = s / TextureSize;
	}
}

// SSR : Screen Space Ratio
// SD  : Signed Distance (encoded : Distance / SDR + .5)
// SDR : Signed Distance Ratio
//
// IsoPerimeter : Dilate / Contract the shape
void ComputeSDF_float(float SSR, float SD, float SDR, float isoPerimeter, float softness, out float outAlpha)
{
	softness *= SSR * SDR;
	float d = (SD - 0.5) * SDR;																				// Signed distance to edge, in Texture space
	outAlpha = saturate((d * 2.0 * SSR + 0.5 + isoPerimeter * SDR * SSR + softness * 0.5) / (1.0 + softness));	// Screen pixel coverage (alpha)
}

void ComputeSDF2_float(float SSR, float SD, float SDR, float2 isoPerimeter, float2 softness, out float2 outAlpha)
{
	softness *= SSR * SDR;
	float d = (SD - 0.5f) * SDR;
	outAlpha = saturate((d * 2.0f * SSR + 0.5f + isoPerimeter * SDR * SSR + softness * 0.5) / (1.0 + softness));
}

void ComputeSDF4_float(float SSR, float SD, float SDR, float4 isoPerimeter, float4 softness, out float4 outAlpha)
{
	softness *= SSR * SDR;
	float d = (SD - 0.5f) * SDR;
	outAlpha = saturate((d * 2.0f * SSR + 0.5f + isoPerimeter * SDR * SSR + softness * 0.5) / (1.0 + softness));
}

void ComputeSDF44_float(float SSR, float4 SD, float SDR, float4 isoPerimeter, float4 softness, bool outline, out float4 outAlpha)
{
	softness *= SSR * SDR;
	float4 d = (SD - 0.5f) * SDR;
	if(outline) d.w = max(max(d.x, d.y), d.z);
	outAlpha = saturate((d * 2.0f * SSR + 0.5f + isoPerimeter * SDR * SSR + softness * 0.5) / (1.0 + softness));
}

void Composite_float(float4 overlying, float4 underlying, out float4 outColor)
{
	outColor = BlendARGB(overlying, underlying);
}

// Face only
void Layer1_float(float alpha, float4 color0, out float4 outColor)
{
	color0.a *= alpha;
	outColor = color0;
}

// Face + 1 Outline
void Layer2_float(float2 alpha, float4 color0, float4 color1, out float4 outColor)
{
	color1.a *= alpha.y;
	color0.rgb *= color0.a; color1.rgb *= color1.a;
	outColor = lerp(color1, color0, alpha.x);
	outColor.rgb /= outColor.a;
}

// Face + 3 Outline
void Layer4_float(float4 alpha, float4 color0, float4 color1, float4 color2, float4 color3, out float4 outColor)
{
	color3.a *= alpha.w;
	color0.rgb *= color0.a; color1.rgb *= color1.a; color2.rgb *= color2.a; color3.rgb *= color3.a;
	outColor = lerp(lerp(lerp(color3, color2, alpha.z), color1, alpha.y), color0, alpha.x);
	outColor.rgb /= outColor.a;
}
