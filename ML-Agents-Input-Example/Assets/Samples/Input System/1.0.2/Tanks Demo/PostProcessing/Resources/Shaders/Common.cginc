#ifndef __COMMON__
#define __COMMON__

#include "UnityCG.cginc"

// Mobile: use RGBM instead of float/half RGB
#define USE_RGBM defined(SHADER_API_MOBILE)

#define MOBILE_OR_CONSOLE (defined(SHADER_API_MOBILE) || defined(SHADER_API_PSSL) || defined(SHADER_API_XBOXONE) || defined(SHADER_API_WIIU))

#if defined(SHADER_API_PSSL)
// No support for sampler2D_half on PS4 in 5.4
#define sampler2D_half sampler2D_float
#endif

// -----------------------------------------------------------------------------
// Uniforms

sampler2D _MainTex;
float4 _MainTex_TexelSize;
float4 _MainTex_ST;

// -----------------------------------------------------------------------------
// Vertex shaders

struct AttributesDefault
{
    float4 vertex : POSITION;
    float4 texcoord : TEXCOORD0;
};

struct VaryingsDefault
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float2 uvSPR : TEXCOORD1; // Single Pass Stereo UVs
};

VaryingsDefault VertDefault(AttributesDefault v)
{
    VaryingsDefault o;
    o.pos = UnityObjectToClipPos(v.vertex);
    o.uv = v.texcoord.xy;
    o.uvSPR = UnityStereoScreenSpaceUVAdjust(v.texcoord.xy, _MainTex_ST);
    return o;
}

// -----------------------------------------------------------------------------
// Maths stuff

#define HALF_MAX        65504.0
#define EPSILON         1.0e-4
#define UNITY_PI_2      (UNITY_PI * 2.0)

inline half Min3(half3 x) { return min(x.x, min(x.y, x.z)); }
inline half Min3(half x, half y, half z) { return min(x, min(y, z)); }

inline half Max3(half3 x) { return max(x.x, max(x.y, x.z)); }
inline half Max3(half x, half y, half z) { return max(x, max(y, z)); }

inline half Min4(half4 x) { return min(x.x, min(x.y, min(x.z, x.w))); }
inline half Min4(half x, half y, half z, half w) { return min(x, min(y, min(z, w))); }

inline half Max4(half4 x) { return max(x.x, max(x.y, max(x.z, x.w))); }
inline half Max4(half x, half y, half z, half w) { return max(x, max(y, min(z, w))); }

inline half  Pow2(half  x) { return x * x; }
inline half2 Pow2(half2 x) { return x * x; }
inline half3 Pow2(half3 x) { return x * x; }
inline half4 Pow2(half4 x) { return x * x; }

inline half  Pow3(half  x) { return x * x * x; }
inline half2 Pow3(half2 x) { return x * x * x; }
inline half3 Pow3(half3 x) { return x * x * x; }
inline half4 Pow3(half4 x) { return x * x * x; }

#ifndef UNITY_STANDARD_BRDF_INCLUDED
inline half  Pow4(half  x) { return x * x * x * x; }
inline half2 Pow4(half2 x) { return x * x * x * x; }
inline half3 Pow4(half3 x) { return x * x * x * x; }
inline half4 Pow4(half4 x) { return x * x * x * x; }
#endif

// Returns the largest vector of v1 and v2
inline half2 MaxV(half2 v1, half2 v2) { return dot(v1, v1) < dot(v2, v2) ? v2 : v1; }
inline half3 MaxV(half3 v1, half3 v2) { return dot(v1, v1) < dot(v2, v2) ? v2 : v1; }
inline half4 MaxV(half4 v1, half4 v2) { return dot(v1, v1) < dot(v2, v2) ? v2 : v1; }

// Clamp HDR value within a safe range
inline half  SafeHDR(half  c) { return min(c, HALF_MAX); }
inline half2 SafeHDR(half2 c) { return min(c, HALF_MAX); }
inline half3 SafeHDR(half3 c) { return min(c, HALF_MAX); }
inline half4 SafeHDR(half4 c) { return min(c, HALF_MAX); }

// Compatibility function
#if (SHADER_TARGET < 50 && !defined(SHADER_API_PSSL))
float rcp(float value)
{
    return 1.0 / value;
}
#endif

// Tonemapper from http://gpuopen.com/optimized-reversible-tonemapper-for-resolve/
float4 FastToneMap(in float4 color)
{
    return float4(color.rgb * rcp(Max3(color.rgb) + 1.), color.a);
}

float4 FastToneMap(in float4 color, in float weight)
{
    return float4(color.rgb * rcp(weight * Max3(color.rgb) + 1.), color.a);
}

float4 FastToneUnmap(in float4 color)
{
    return float4(color.rgb * rcp(1. - Max3(color.rgb)), color.a);
}

// Interleaved gradient function from Jimenez 2014 http://goo.gl/eomGso
float GradientNoise(float2 uv)
{
    uv = floor(uv * _ScreenParams.xy);
    float f = dot(float2(0.06711056, 0.00583715), uv);
    return frac(52.9829189 * frac(f));
}

// Z buffer depth to linear 0-1 depth
// Handles orthographic projection correctly
float LinearizeDepth(float z)
{
    float isOrtho = unity_OrthoParams.w;
    float isPers = 1.0 - unity_OrthoParams.w;
    z *= _ZBufferParams.x;
    return (1.0 - isOrtho * z) / (isPers * z + _ZBufferParams.y);
}

// -----------------------------------------------------------------------------
// RGBM encoding/decoding

half4 EncodeHDR(float3 rgb)
{
#if USE_RGBM
    rgb *= 1.0 / 8.0;
    float m = max(max(rgb.r, rgb.g), max(rgb.b, 1e-6));
    m = ceil(m * 255.0) / 255.0;
    return half4(rgb / m, m);
#else
    return half4(rgb, 0.0);
#endif
}

float3 DecodeHDR(half4 rgba)
{
#if USE_RGBM
    return rgba.rgb * rgba.a * 8.0;
#else
    return rgba.rgb;
#endif
}

#endif // __COMMON__
