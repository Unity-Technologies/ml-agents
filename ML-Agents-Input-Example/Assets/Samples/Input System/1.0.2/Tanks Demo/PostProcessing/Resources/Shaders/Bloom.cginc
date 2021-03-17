#ifndef __BLOOM__
#define __BLOOM__

#include "Common.cginc"

// Brightness function
half Brightness(half3 c)
{
    return Max3(c);
}

// 3-tap median filter
half3 Median(half3 a, half3 b, half3 c)
{
    return a + b + c - min(min(a, b), c) - max(max(a, b), c);
}

// Downsample with a 4x4 box filter
half3 DownsampleFilter(sampler2D tex, float2 uv, float2 texelSize)
{
    float4 d = texelSize.xyxy * float4(-1.0, -1.0, 1.0, 1.0);

    half3 s;
    s = DecodeHDR(tex2D(tex, uv + d.xy));
    s += DecodeHDR(tex2D(tex, uv + d.zy));
    s += DecodeHDR(tex2D(tex, uv + d.xw));
    s += DecodeHDR(tex2D(tex, uv + d.zw));

    return s * (1.0 / 4.0);
}

// Downsample with a 4x4 box filter + anti-flicker filter
half3 DownsampleAntiFlickerFilter(sampler2D tex, float2 uv, float2 texelSize)
{
    float4 d = texelSize.xyxy * float4(-1.0, -1.0, 1.0, 1.0);

    half3 s1 = DecodeHDR(tex2D(tex, uv + d.xy));
    half3 s2 = DecodeHDR(tex2D(tex, uv + d.zy));
    half3 s3 = DecodeHDR(tex2D(tex, uv + d.xw));
    half3 s4 = DecodeHDR(tex2D(tex, uv + d.zw));

    // Karis's luma weighted average (using brightness instead of luma)
    half s1w = 1.0 / (Brightness(s1) + 1.0);
    half s2w = 1.0 / (Brightness(s2) + 1.0);
    half s3w = 1.0 / (Brightness(s3) + 1.0);
    half s4w = 1.0 / (Brightness(s4) + 1.0);
    half one_div_wsum = 1.0 / (s1w + s2w + s3w + s4w);

    return (s1 * s1w + s2 * s2w + s3 * s3w + s4 * s4w) * one_div_wsum;
}

half3 UpsampleFilter(sampler2D tex, float2 uv, float2 texelSize, float sampleScale)
{
#if MOBILE_OR_CONSOLE
    // 4-tap bilinear upsampler
    float4 d = texelSize.xyxy * float4(-1.0, -1.0, 1.0, 1.0) * (sampleScale * 0.5);

    half3 s;
    s =  DecodeHDR(tex2D(tex, uv + d.xy));
    s += DecodeHDR(tex2D(tex, uv + d.zy));
    s += DecodeHDR(tex2D(tex, uv + d.xw));
    s += DecodeHDR(tex2D(tex, uv + d.zw));

    return s * (1.0 / 4.0);
#else
    // 9-tap bilinear upsampler (tent filter)
    float4 d = texelSize.xyxy * float4(1.0, 1.0, -1.0, 0.0) * sampleScale;

    half3 s;
    s =  DecodeHDR(tex2D(tex, uv - d.xy));
    s += DecodeHDR(tex2D(tex, uv - d.wy)) * 2.0;
    s += DecodeHDR(tex2D(tex, uv - d.zy));

    s += DecodeHDR(tex2D(tex, uv + d.zw)) * 2.0;
    s += DecodeHDR(tex2D(tex, uv))        * 4.0;
    s += DecodeHDR(tex2D(tex, uv + d.xw)) * 2.0;

    s += DecodeHDR(tex2D(tex, uv + d.zy));
    s += DecodeHDR(tex2D(tex, uv + d.wy)) * 2.0;
    s += DecodeHDR(tex2D(tex, uv + d.xy));

    return s * (1.0 / 16.0);
#endif
}

#endif // __BLOOM__
