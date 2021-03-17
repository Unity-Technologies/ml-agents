#ifndef __COLOR_GRADING__
#define __COLOR_GRADING__

#include "ACES.cginc"
#include "Common.cginc"

// Set to 1 to use more precise but more expensive log/linear conversions. I haven't found a proper
// use case for the high precision version yet so I'm leaving this to 0.
#define COLOR_GRADING_PRECISE_LOG 0

//
// Alexa LogC converters (El 1000)
// See http://www.vocas.nl/webfm_send/964
// It's a good fit to store HDR values in log as the range is pretty wide (1 maps to ~58.85666) and
// is quick enough to compute.
//
struct ParamsLogC
{
    half cut;
    half a, b, c, d, e, f;
};

static const ParamsLogC LogC =
{
    0.011361, // cut
    5.555556, // a
    0.047996, // b
    0.244161, // c
    0.386036, // d
    5.301883, // e
    0.092819  // f
};

half LinearToLogC_Precise(half x)
{
    half o;
    if (x > LogC.cut)
        o = LogC.c * log10(LogC.a * x + LogC.b) + LogC.d;
    else
        o = LogC.e * x + LogC.f;
    return o;
}

half3 LinearToLogC(half3 x)
{
#if COLOR_GRADING_PRECISE_LOG
    return half3(
        LinearToLogC_Precise(x.x),
        LinearToLogC_Precise(x.y),
        LinearToLogC_Precise(x.z)
    );
#else
    return LogC.c * log10(LogC.a * x + LogC.b) + LogC.d;
#endif
}

half LogCToLinear_Precise(half x)
{
    half o;
    if (x > LogC.e * LogC.cut + LogC.f)
        o = (pow(10.0, (x - LogC.d) / LogC.c) - LogC.b) / LogC.a;
    else
        o = (x - LogC.f) / LogC.e;
    return o;
}

half3 LogCToLinear(half3 x)
{
#if COLOR_GRADING_PRECISE_LOG
    return half3(
        LogCToLinear_Precise(x.x),
        LogCToLinear_Precise(x.y),
        LogCToLinear_Precise(x.z)
    );
#else
    return (pow(10.0, (x - LogC.d) / LogC.c) - LogC.b) / LogC.a;
#endif
}

//
// White balance
// Recommended workspace: ACEScg (linear)
//
static const half3x3 LIN_2_LMS_MAT = {
    3.90405e-1, 5.49941e-1, 8.92632e-3,
    7.08416e-2, 9.63172e-1, 1.35775e-3,
    2.31082e-2, 1.28021e-1, 9.36245e-1
};

static const half3x3 LMS_2_LIN_MAT = {
     2.85847e+0, -1.62879e+0, -2.48910e-2,
    -2.10182e-1,  1.15820e+0,  3.24281e-4,
    -4.18120e-2, -1.18169e-1,  1.06867e+0
};

half3 WhiteBalance(half3 c, half3 balance)
{
    half3 lms = mul(LIN_2_LMS_MAT, c);
    lms *= balance;
    return mul(LMS_2_LIN_MAT, lms);
}

//
// Luminance (Rec.709 primaries according to ACES specs)
//
half AcesLuminance(half3 c)
{
    return dot(c, half3(0.2126, 0.7152, 0.0722));
}

//
// Offset, Power, Slope (ASC-CDL)
// Works in Log & Linear. Results will be different but still correct.
//
half3 OffsetPowerSlope(half3 c, half3 offset, half3 power, half3 slope)
{
    half3 so = c * slope + offset;
    so = so > (0.0).xxx ? pow(so, power) : so;
    return so;
}

//
// Lift, Gamma (pre-inverted), Gain
// Recommended workspace: ACEScg (linear)
//
half3 LiftGammaGain(half3 c, half3 lift, half3 invgamma, half3 gain)
{
    //return gain * (lift * (1.0 - c) + pow(max(c, kEpsilon), invgamma));
    //return pow(gain * (c + lift * (1.0 - c)), invgamma);

    half3 power = invgamma;
    half3 offset = lift * gain;
    half3 slope = ((1.0).xxx - lift) * gain;
    return OffsetPowerSlope(c, offset, power, slope);
}

//
// Saturation (should be used after offset/power/slope)
// Recommended workspace: ACEScc (log)
// Optimal range: [0.0, 2.0]
//
half3 Saturation(half3 c, half sat)
{
    half luma = AcesLuminance(c);
    return luma.xxx + sat * (c - luma.xxx);
}

//
// Basic contrast curve
// Recommended workspace: ACEScc (log)
// Optimal range: [0.0, 2.0]
//
half3 ContrastLog(half3 c, half con)
{
    return (c - ACEScc_MIDGRAY) * con + ACEScc_MIDGRAY;
}

//
// Hue, Saturation, Value
// Ranges:
//  Hue [0.0, 1.0]
//  Sat [0.0, 1.0]
//  Lum [0.0, HALF_MAX]
//
half3 RgbToHsv(half3 c)
{
    half4 K = half4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    half4 p = lerp(half4(c.bg, K.wz), half4(c.gb, K.xy), step(c.b, c.g));
    half4 q = lerp(half4(p.xyw, c.r), half4(c.r, p.yzx), step(p.x, c.r));
    half d = q.x - min(q.w, q.y);
    half e = EPSILON;
    return half3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

half3 HsvToRgb(half3 c)
{
    half4 K = half4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    half3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

half RotateHue(half value, half low, half hi)
{
    return (value < low)
            ? value + hi
            : (value > hi)
                ? value - hi
                : value;
}

//
// Remaps Y/R/G/B values
//
half3 YrgbCurve(half3 c, sampler2D curveTex)
{
    const float kHalfPixel = (1.0 / 128.0) / 2.0;

    // Y
    c += kHalfPixel.xxx;
    float mr = tex2D(curveTex, float2(c.r, 0.75)).a;
    float mg = tex2D(curveTex, float2(c.g, 0.75)).a;
    float mb = tex2D(curveTex, float2(c.b, 0.75)).a;
    c = saturate(float3(mr, mg, mb));

    // RGB
    c += kHalfPixel.xxx;
    float r = tex2D(curveTex, float2(c.r, 0.75)).r;
    float g = tex2D(curveTex, float2(c.g, 0.75)).g;
    float b = tex2D(curveTex, float2(c.b, 0.75)).b;
    return saturate(half3(r, g, b));
}

//
// (X) Hue VS Hue - Remaps hue on a curve according to the current hue
//      Input is Hue [0.0, 1.0]
//      Output is Hue [0.0, 1.0]
//
half SecondaryHueHue(half hue, sampler2D curveTex)
{
    half offset = saturate(tex2D(curveTex, half2(hue, 0.25)).x) - 0.5;
    hue += offset;
    hue = RotateHue(hue, 0.0, 1.0);
    return hue;
}

//
// (Y) Hue VS Saturation - Remaps saturation on a curve according to the current hue
//      Input is Hue [0.0, 1.0]
//      Output is Saturation multiplier [0.0, 2.0]
//
half SecondaryHueSat(half hue, sampler2D curveTex)
{
    return saturate(tex2D(curveTex, half2(hue, 0.25)).y) * 2.0;
}

//
// (Z) Saturation VS Saturation - Remaps saturation on a curve according to the current saturation
//      Input is Saturation [0.0, 1.0]
//      Output is Saturation multiplier [0.0, 2.0]
//
half SecondarySatSat(half sat, sampler2D curveTex)
{
    return saturate(tex2D(curveTex, half2(sat, 0.25)).z) * 2.0;
}

//
// (W) Luminance VS Saturation - Remaps saturation on a curve according to the current luminance
//      Input is Luminance [0.0, 1.0]
//      Output is Saturation multiplier [0.0, 2.0]
//
half SecondaryLumSat(half lum, sampler2D curveTex)
{
    return saturate(tex2D(curveTex, half2(lum, 0.25)).w) * 2.0;
}

//
// Channel mixing (same as Photoshop's and DaVinci's Resolve)
// Recommended workspace: ACEScg (linear)
//      Input mixers should be in range [-2.0;2.0]
//
half3 ChannelMixer(half3 c, half3 red, half3 green, half3 blue)
{
    return half3(
        dot(c, red),
        dot(c, green),
        dot(c, blue)
    );
}

//
// LUT grading
// scaleOffset = (1 / lut_width, 1 / lut_height, lut_height - 1)
//
half3 ApplyLut2d(sampler2D tex, half3 uvw, half3 scaleOffset)
{
    // Strip format where `height = sqrt(width)`
    uvw.z *= scaleOffset.z;
    half shift = floor(uvw.z);
    uvw.xy = uvw.xy * scaleOffset.z * scaleOffset.xy + scaleOffset.xy * 0.5;
    uvw.x += shift * scaleOffset.y;
    uvw.xyz = lerp(tex2D(tex, uvw.xy).rgb, tex2D(tex, uvw.xy + half2(scaleOffset.y, 0)).rgb, uvw.z - shift);
    return uvw;
}

half3 ApplyLut3d(sampler3D tex, half3 uvw)
{
    return tex3D(tex, uvw).rgb;
}

#endif // __COLOR_GRADING__
