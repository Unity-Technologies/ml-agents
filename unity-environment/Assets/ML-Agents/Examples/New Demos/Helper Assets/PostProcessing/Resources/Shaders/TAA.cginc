#ifndef __TAA__
#define __TAA__

#pragma only_renderers ps4 xboxone d3d11 d3d9 xbox360 opengl glcore
#pragma exclude_renderers gles

#include "UnityCG.cginc"
#include "Common.cginc"

// -----------------------------------------------------------------------------
// Solver

#define TAA_USE_STABLE_BUT_GHOSTY_VARIANT 0

#if !defined(TAA_DILATE_MOTION_VECTOR_SAMPLE)
    #define TAA_DILATE_MOTION_VECTOR_SAMPLE 1
#endif

#define TAA_FRAGMENT_MOTION_HISTORY_DECAY 0.85

#define TAA_FINAL_BLEND_STATIC_FACTOR _FinalBlendParameters.x
#define TAA_FINAL_BLEND_DYNAMIC_FACTOR _FinalBlendParameters.y
#define TAA_MOTION_AMPLIFICATION _FinalBlendParameters.z

struct VaryingsSolver
{
    float4 vertex : SV_POSITION;
    float4 uv : TEXCOORD0; // [xy: _MainTex.uv, zw: _HistoryTex.uv]
};

struct OutputSolver
{
    float4 destination : SV_Target0;
    float4 history : SV_Target1;
};

sampler2D _HistoryTex;

sampler2D _CameraMotionVectorsTexture;
sampler2D _CameraDepthTexture;

float4 _HistoryTex_TexelSize;
float4 _CameraDepthTexture_TexelSize;

float2 _Jitter;
float4 _SharpenParameters;
float4 _FinalBlendParameters;

VaryingsSolver VertSolver(AttributesDefault input)
{
    VaryingsSolver output;

    float4 vertex = UnityObjectToClipPos(input.vertex);

    output.vertex = vertex;
    output.uv = input.texcoord.xyxy;

#if UNITY_UV_STARTS_AT_TOP
    if (_MainTex_TexelSize.y < 0)
        output.uv.y = 1.0 - input.texcoord.y;
#endif

    return output;
}

float2 GetClosestFragment(float2 uv)
{
    const float2 k = _CameraDepthTexture_TexelSize.xy;
    const float4 neighborhood = float4(
        SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv - k),
        SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv + float2(k.x, -k.y)),
        SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv + float2(-k.x, k.y)),
        SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv + k)
        );

#if defined(UNITY_REVERSED_Z)
    #define COMPARE_DEPTH(a, b) step(b, a)
#else
    #define COMPARE_DEPTH(a, b) step(a, b)
#endif

    float3 result = float3(0.0, 0.0, SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv));
    result = lerp(result, float3(-1.0, -1.0, neighborhood.x), COMPARE_DEPTH(neighborhood.x, result.z));
    result = lerp(result, float3( 1.0, -1.0, neighborhood.y), COMPARE_DEPTH(neighborhood.y, result.z));
    result = lerp(result, float3(-1.0,  1.0, neighborhood.z), COMPARE_DEPTH(neighborhood.z, result.z));
    result = lerp(result, float3( 1.0,  1.0, neighborhood.w), COMPARE_DEPTH(neighborhood.w, result.z));

    return (uv + result.xy * k);
}

// Adapted from Playdead's TAA implementation
// https://github.com/playdeadgames/temporal
float4 ClipToAABB(float4 color, float p, float3 minimum, float3 maximum)
{
    // note: only clips towards aabb center (but fast!)
    float3 center  = 0.5 * (maximum + minimum);
    float3 extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    float4 offset = color - float4(center, p);
    float3 repeat = abs(offset.xyz / extents);

    repeat.x = max(repeat.x, max(repeat.y, repeat.z));

    if (repeat.x > 1.0)
    {
        // `color` is not intersecting (nor inside) the AABB; it's clipped to the closest extent
        return float4(center, p) + offset / repeat.x;
    }
    else
    {
        // `color` is intersecting (or inside) the AABB.

        // Note: for whatever reason moving this return statement from this else into a higher
        // scope makes the NVIDIA drivers go beyond bonkers
        return color;
    }
}

OutputSolver FragSolver(VaryingsSolver input)
{
#if TAA_DILATE_MOTION_VECTOR_SAMPLE
    float2 motion = tex2D(_CameraMotionVectorsTexture, GetClosestFragment(input.uv.zw)).xy;
#else
    // Don't dilate in ortho !
    float2 motion = tex2D(_CameraMotionVectorsTexture, input.uv.zw).xy;
#endif

    const float2 k = _MainTex_TexelSize.xy;
    float2 uv = input.uv.xy;

#if UNITY_UV_STARTS_AT_TOP
    uv -= _MainTex_TexelSize.y < 0 ? _Jitter * float2(1.0, -1.0) : _Jitter;
#else
    uv -= _Jitter;
#endif

    float4 color = tex2D(_MainTex, uv);

    float4 topLeft = tex2D(_MainTex, uv - k * 0.5);
    float4 bottomRight = tex2D(_MainTex, uv + k * 0.5);

    float4 corners = 4.0 * (topLeft + bottomRight) - 2.0 * color;

    // Sharpen output
    color += (color - (corners * 0.166667)) * 2.718282 * _SharpenParameters.x;
    color = max(0.0, color);

    // Tonemap color and history samples
    float4 average = FastToneMap((corners + color) * 0.142857);

    topLeft = FastToneMap(topLeft);
    bottomRight = FastToneMap(bottomRight);

    color = FastToneMap(color);

    float4 history = tex2D(_HistoryTex, input.uv.zw - motion);

// Only use this variant for arch viz or scenes that don't have any animated objects (camera animation is fine)
#if TAA_USE_STABLE_BUT_GHOSTY_VARIANT
    float4 luma = float4(Luminance(topLeft.rgb), Luminance(bottomRight.rgb), Luminance(average.rgb), Luminance(color.rgb));
    float nudge = lerp(6.28318530718, 0.5, saturate(2.0 * history.a)) * max(abs(luma.z - luma.w), abs(luma.x - luma.y));

    float4 minimum = lerp(bottomRight, topLeft, step(luma.x, luma.y)) - nudge;
    float4 maximum = lerp(topLeft, bottomRight, step(luma.x, luma.y)) + nudge;
#else
    float2 luma = float2(Luminance(average.rgb), Luminance(color.rgb));
    float nudge = 4.0 * abs(luma.x - luma.y);

    float4 minimum = min(bottomRight, topLeft) - nudge;
    float4 maximum = max(topLeft, bottomRight) + nudge;
#endif

    history = FastToneMap(history);

    // Clip history samples
    history = ClipToAABB(history, history.a, minimum.xyz, maximum.xyz);

    // Store fragment motion history
    color.a = saturate(smoothstep(0.002 * _MainTex_TexelSize.z, 0.0035 * _MainTex_TexelSize.z, length(motion)));

    // Blend method
    float weight = clamp(lerp(TAA_FINAL_BLEND_STATIC_FACTOR, TAA_FINAL_BLEND_DYNAMIC_FACTOR,
        length(motion) * TAA_MOTION_AMPLIFICATION), TAA_FINAL_BLEND_DYNAMIC_FACTOR, TAA_FINAL_BLEND_STATIC_FACTOR);

    color = FastToneUnmap(lerp(color, history, weight));

    OutputSolver output;

    output.destination = color;
    color.a *= TAA_FRAGMENT_MOTION_HISTORY_DECAY;

    output.history = color;

    return output;
}

// -----------------------------------------------------------------------------
// Alpha clearance

float4 FragAlphaClear(VaryingsDefault input) : SV_Target
{
    return float4(tex2D(_MainTex, input.uv).rgb, 0.0);
}

#endif // __TAA__
