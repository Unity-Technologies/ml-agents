#ifndef __TONEMAPPING__
#define __TONEMAPPING__

#include "ACES.cginc"

// Set to 1 to use the full reference ACES tonemapper. This should only be used for research
// purposes and it's quite heavy and generally overkill.
#define TONEMAPPING_USE_FULL_ACES 0

//
// Neutral tonemapping (Hable/Hejl/Frostbite)
// Input is linear RGB
//
half3 NeutralCurve(half3 x, half a, half b, half c, half d, half e, half f)
{
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

half3 NeutralTonemap(half3 x, half4 params1, half4 params2)
{
    // ACES supports negative color values and WILL output negative values when coming from ACES or ACEScg
    // Make sure negative channels are clamped to 0.0 as this neutral tonemapper can't deal with them properly
    x = max((0.0).xxx, x);

    // Tonemap
    half a = params1.x;
    half b = params1.y;
    half c = params1.z;
    half d = params1.w;
    half e = params2.x;
    half f = params2.y;
    half whiteLevel = params2.z;
    half whiteClip = params2.w;

    half3 whiteScale = (1.0).xxx / NeutralCurve(whiteLevel, a, b, c, d, e, f);
    x = NeutralCurve(x * whiteScale, a, b, c, d, e, f);
    x *= whiteScale;

    // Post-curve white point adjustment
    x /= whiteClip.xxx;

    return x;
}

//
// Filmic tonemapping (ACES fitting, unless TONEMAPPING_USE_FULL_ACES is set to 1)
// Input is ACES2065-1 (AP0 w/ linear encoding)
//
half3 FilmicTonemap(half3 aces)
{
#if TONEMAPPING_USE_FULL_ACES

    half3 oces = RRT(aces);
    half3 odt = ODT_RGBmonitor_100nits_dim(oces);
    return odt;

#else

    // --- Glow module --- //
    half saturation = rgb_2_saturation(aces);
    half ycIn = rgb_2_yc(aces);
    half s = sigmoid_shaper((saturation - 0.4) / 0.2);
    half addedGlow = 1.0 + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
    aces *= addedGlow;

    // --- Red modifier --- //
    half hue = rgb_2_hue(aces);
    half centeredHue = center_hue(hue, RRT_RED_HUE);
    half hueWeight;
    {
        //hueWeight = cubic_basis_shaper(centeredHue, RRT_RED_WIDTH);
        hueWeight = Pow2(smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centeredHue / RRT_RED_WIDTH)));
    }

    aces.r += hueWeight * saturation * (RRT_RED_PIVOT - aces.r) * (1.0 - RRT_RED_SCALE);

    // --- ACES to RGB rendering space --- //
    half3 acescg = max(0.0, ACES_to_ACEScg(aces));

    // --- Global desaturation --- //
    //acescg = mul(RRT_SAT_MAT, acescg);
    acescg = lerp(dot(acescg, AP1_RGB2Y).xxx, acescg, RRT_SAT_FACTOR.xxx);

    // Luminance fitting of *RRT.a1.0.3 + ODT.Academy.RGBmonitor_100nits_dim.a1.0.3*.
    // https://github.com/colour-science/colour-unity/blob/master/Assets/Colour/Notebooks/CIECAM02_Unity.ipynb
    // RMSE: 0.0012846272106
    const half a = 278.5085;
    const half b = 10.7772;
    const half c = 293.6045;
    const half d = 88.7122;
    const half e = 80.6889;
    half3 x = acescg;
    half3 rgbPost = (x * (a * x + b)) / (x * (c * x + d) + e);

    // Scale luminance to linear code value
    // half3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

    // Apply gamma adjustment to compensate for dim surround
    half3 linearCV = darkSurround_to_dimSurround(rgbPost);

    // Apply desaturation to compensate for luminance difference
    //linearCV = mul(ODT_SAT_MAT, color);
    linearCV = lerp(dot(linearCV, AP1_RGB2Y).xxx, linearCV, ODT_SAT_FACTOR.xxx);

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    // Apply CAT from ACES white point to assumed observer adapted white point
    XYZ = mul(D60_2_D65_CAT, XYZ);

    // CIE XYZ to display primaries
    linearCV = mul(XYZ_2_REC709_MAT, XYZ);

    return linearCV;

#endif
}

#endif // __TONEMAPPING__
