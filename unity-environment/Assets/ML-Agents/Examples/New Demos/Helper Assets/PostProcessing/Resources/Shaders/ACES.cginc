#ifndef __ACES__
#define __ACES__

/**
 * https://github.com/ampas/aces-dev
 *
 * Academy Color Encoding System (ACES) software and tools are provided by the
 * Academy under the following terms and conditions: A worldwide, royalty-free,
 * non-exclusive right to copy, modify, create derivatives, and use, in source and
 * binary forms, is hereby granted, subject to acceptance of this license.
 *
 * Copyright 2015 Academy of Motion Picture Arts and Sciences (A.M.P.A.S.).
 * Portions contributed by others as indicated. All rights reserved.
 *
 * Performance of any of the aforementioned acts indicates acceptance to be bound
 * by the following terms and conditions:
 *
 * * Copies of source code, in whole or in part, must retain the above copyright
 * notice, this list of conditions and the Disclaimer of Warranty.
 *
 * * Use in binary form must retain the above copyright notice, this list of
 * conditions and the Disclaimer of Warranty in the documentation and/or other
 * materials provided with the distribution.
 *
 * * Nothing in this license shall be deemed to grant any rights to trademarks,
 * copyrights, patents, trade secrets or any other intellectual property of
 * A.M.P.A.S. or any contributors, except as expressly stated herein.
 *
 * * Neither the name "A.M.P.A.S." nor the name of any other contributors to this
 * software may be used to endorse or promote products derivative of or based on
 * this software without express prior written permission of A.M.P.A.S. or the
 * contributors, as appropriate.
 *
 * This license shall be construed pursuant to the laws of the State of
 * California, and any disputes related thereto shall be subject to the
 * jurisdiction of the courts therein.
 *
 * Disclaimer of Warranty: THIS SOFTWARE IS PROVIDED BY A.M.P.A.S. AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
 * NON-INFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL A.M.P.A.S., OR ANY
 * CONTRIBUTORS OR DISTRIBUTORS, BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, RESITUTIONARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * WITHOUT LIMITING THE GENERALITY OF THE FOREGOING, THE ACADEMY SPECIFICALLY
 * DISCLAIMS ANY REPRESENTATIONS OR WARRANTIES WHATSOEVER RELATED TO PATENT OR
 * OTHER INTELLECTUAL PROPERTY RIGHTS IN THE ACADEMY COLOR ENCODING SYSTEM, OR
 * APPLICATIONS THEREOF, HELD BY PARTIES OTHER THAN A.M.P.A.S.,WHETHER DISCLOSED OR
 * UNDISCLOSED.
 */

//#define CUSTOM_WHITE_POINT

/*
    Basic usage :

    half4 color = tex2D(_MainTex, i.uv);
    half3 aces = unity_to_ACES(color.rgb);
    half3 oces = RRT(aces);
    half3 odt = ODT_RGBmonitor_100nits_dim(oces);
    return half4(odt, color.a);

    If you want to customize the white point, uncomment the previous define and set uniforms accordingly:

    float whitePoint = 48f; // Default ACES value
    material.SetFloat("CINEMA_WHITE", whitePoint);
    material.SetFloat("CINEMA_DARK", whitePoint / 2400f);
 */

#include "Common.cginc"

#define ACEScc_MAX      1.4679964
#define ACEScc_MIDGRAY  0.4135884

//
// Precomputed matrices (pre-transposed)
// See https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
//
static const half3x3 sRGB_2_AP0 = {
    0.4397010, 0.3829780, 0.1773350,
    0.0897923, 0.8134230, 0.0967616,
    0.0175440, 0.1115440, 0.8707040
};

static const half3x3 sRGB_2_AP1 = {
    0.61319, 0.33951, 0.04737,
    0.07021, 0.91634, 0.01345,
    0.02062, 0.10957, 0.86961
};

static const half3x3 AP0_2_sRGB = {
    2.52169, -1.13413, -0.38756,
    -0.27648, 1.37272, -0.09624,
    -0.01538, -0.15298, 1.16835,
};

static const half3x3 AP1_2_sRGB = {
    1.70505, -0.62179, -0.08326,
    -0.13026, 1.14080, -0.01055,
    -0.02400, -0.12897, 1.15297,
};

static const half3x3 AP0_2_AP1_MAT = {
     1.4514393161, -0.2365107469, -0.2149285693,
    -0.0765537734,  1.1762296998, -0.0996759264,
     0.0083161484, -0.0060324498,  0.9977163014
};

static const half3x3 AP1_2_AP0_MAT = {
     0.6954522414, 0.1406786965, 0.1638690622,
     0.0447945634, 0.8596711185, 0.0955343182,
    -0.0055258826, 0.0040252103, 1.0015006723
};

static const half3x3 AP1_2_XYZ_MAT = {
     0.6624541811, 0.1340042065, 0.1561876870,
     0.2722287168, 0.6740817658, 0.0536895174,
    -0.0055746495, 0.0040607335, 1.0103391003
};

static const half3x3 XYZ_2_AP1_MAT = {
     1.6410233797, -0.3248032942, -0.2364246952,
    -0.6636628587,  1.6153315917,  0.0167563477,
     0.0117218943, -0.0082844420,  0.9883948585
};

static const half3x3 XYZ_2_REC709_MAT = {
     3.2409699419, -1.5373831776, -0.4986107603,
    -0.9692436363,  1.8759675015,  0.0415550574,
     0.0556300797, -0.2039769589,  1.0569715142
};

static const half3x3 XYZ_2_REC2020_MAT = {
     1.7166511880, -0.3556707838, -0.2533662814,
    -0.6666843518,  1.6164812366,  0.0157685458,
     0.0176398574, -0.0427706133,  0.9421031212
};

static const half3x3 XYZ_2_DCIP3_MAT = {
     2.7253940305, -1.0180030062, -0.4401631952,
    -0.7951680258,  1.6897320548,  0.0226471906,
     0.0412418914, -0.0876390192,  1.1009293786
};

static const half3 AP1_RGB2Y = half3(0.272229, 0.674082, 0.0536895);

static const half3x3 RRT_SAT_MAT = {
    0.9708890, 0.0269633, 0.00214758,
    0.0108892, 0.9869630, 0.00214758,
    0.0108892, 0.0269633, 0.96214800
};

static const half3x3 ODT_SAT_MAT = {
    0.949056, 0.0471857, 0.00375827,
    0.019056, 0.9771860, 0.00375827,
    0.019056, 0.0471857, 0.93375800
};

static const half3x3 D60_2_D65_CAT = {
     0.98722400, -0.00611327, 0.0159533,
    -0.00759836,  1.00186000, 0.0053302,
     0.00307257, -0.00509595, 1.0816800
};

//
// Unity to ACES
//
// converts Unity raw (sRGB primaries) to
//          ACES2065-1 (AP0 w/ linear encoding)
//
half3 unity_to_ACES(half3 x)
{
    x = mul(sRGB_2_AP0, x);
    return x;
}

//
// ACES to Unity
//
// converts ACES2065-1 (AP0 w/ linear encoding)
//          Unity raw (sRGB primaries) to
//
half3 ACES_to_unity(half3 x)
{
    x = mul(AP0_2_sRGB, x);
    return x;
}

//
// Unity to ACEScg
//
// converts Unity raw (sRGB primaries) to
//          ACEScg (AP1 w/ linear encoding)
//
half3 unity_to_ACEScg(half3 x)
{
    x = mul(sRGB_2_AP1, x);
    return x;
}

//
// ACEScg to Unity
//
// converts ACEScg (AP1 w/ linear encoding) to
//          Unity raw (sRGB primaries)
//
half3 ACEScg_to_unity(half3 x)
{
    x = mul(AP1_2_sRGB, x);
    return x;
}

//
// ACES Color Space Conversion - ACES to ACEScc
//
// converts ACES2065-1 (AP0 w/ linear encoding) to
//          ACEScc (AP1 w/ logarithmic encoding)
//
// This transform follows the formulas from section 4.4 in S-2014-003
//
half ACES_to_ACEScc(half x)
{
    if (x <= 0.0)
        return -0.35828683; // = (log2(pow(2.0, -15.0) * 0.5) + 9.72) / 17.52
    else if (x < pow(2.0, -15.0))
        return (log2(pow(2.0, -16.0) + x * 0.5) + 9.72) / 17.52;
    else // (x >= pow(2.0, -15.0))
        return (log2(x) + 9.72) / 17.52;
}

half3 ACES_to_ACEScc(half3 x)
{
    x = clamp(x, 0.0, HALF_MAX);

    // x is clamped to [0, HALF_MAX], skip the <= 0 check
    return (x < 0.00003051757) ? (log2(0.00001525878 + x * 0.5) + 9.72) / 17.52 : (log2(x) + 9.72) / 17.52;

    /*
    return half3(
        ACES_to_ACEScc(x.r),
        ACES_to_ACEScc(x.g),
        ACES_to_ACEScc(x.b)
    );
    */
}

//
// ACES Color Space Conversion - ACEScc to ACES
//
// converts ACEScc (AP1 w/ ACESlog encoding) to
//          ACES2065-1 (AP0 w/ linear encoding)
//
// This transform follows the formulas from section 4.4 in S-2014-003
//
half ACEScc_to_ACES(half x)
{
    // TODO: Optimize me
    if (x < -0.3013698630) // (9.72 - 15) / 17.52
        return (pow(2.0, x * 17.52 - 9.72) - pow(2.0, -16.0)) * 2.0;
    else if (x < (log2(HALF_MAX) + 9.72) / 17.52)
        return pow(2.0, x * 17.52 - 9.72);
    else // (x >= (log2(HALF_MAX) + 9.72) / 17.52)
        return HALF_MAX;
}

half3 ACEScc_to_ACES(half3 x)
{
    return half3(
        ACEScc_to_ACES(x.r),
        ACEScc_to_ACES(x.g),
        ACEScc_to_ACES(x.b)
    );
}

//
// ACES Color Space Conversion - ACES to ACEScg
//
// converts ACES2065-1 (AP0 w/ linear encoding) to
//          ACEScg (AP1 w/ linear encoding)
//
half3 ACES_to_ACEScg(half3 x)
{
    return mul(AP0_2_AP1_MAT, x);
}

//
// ACES Color Space Conversion - ACEScg to ACES
//
// converts ACEScg (AP1 w/ linear encoding) to
//          ACES2065-1 (AP0 w/ linear encoding)
//
half3 ACEScg_to_ACES(half3 x)
{
    return mul(AP1_2_AP0_MAT, x);
}

//
// Reference Rendering Transform (RRT)
//
//   Input is ACES
//   Output is OCES
//
half rgb_2_saturation(half3 rgb)
{
    const half TINY = 1e-10;
    half mi = Min3(rgb);
    half ma = Max3(rgb);
    return (max(ma, TINY) - max(mi, TINY)) / max(ma, 1e-2);
}

half rgb_2_yc(half3 rgb)
{
    const half ycRadiusWeight = 1.75;

    // Converts RGB to a luminance proxy, here called YC
    // YC is ~ Y + K * Chroma
    // Constant YC is a cone-shaped surface in RGB space, with the tip on the
    // neutral axis, towards white.
    // YC is normalized: RGB 1 1 1 maps to YC = 1
    //
    // ycRadiusWeight defaults to 1.75, although can be overridden in function
    // call to rgb_2_yc
    // ycRadiusWeight = 1 -> YC for pure cyan, magenta, yellow == YC for neutral
    // of same value
    // ycRadiusWeight = 2 -> YC for pure red, green, blue  == YC for  neutral of
    // same value.

    half r = rgb.x;
    half g = rgb.y;
    half b = rgb.z;
    half chroma = sqrt(b * (b - g) + g * (g - r) + r * (r - b));
    return (b + g + r + ycRadiusWeight * chroma) / 3.0;
}

half rgb_2_hue(half3 rgb)
{
    // Returns a geometric hue angle in degrees (0-360) based on RGB values.
    // For neutral colors, hue is undefined and the function will return a quiet NaN value.
    half hue;
    if (rgb.x == rgb.y && rgb.y == rgb.z)
        hue = 0.0; // RGB triplets where RGB are equal have an undefined hue
    else
        hue = (180.0 / UNITY_PI) * atan2(sqrt(3.0) * (rgb.y - rgb.z), 2.0 * rgb.x - rgb.y - rgb.z);

    if (hue < 0.0) hue = hue + 360.0;

    return hue;
}

half center_hue(half hue, half centerH)
{
    half hueCentered = hue - centerH;
    if (hueCentered < -180.0) hueCentered = hueCentered + 360.0;
    else if (hueCentered > 180.0) hueCentered = hueCentered - 360.0;
    return hueCentered;
}

half sigmoid_shaper(half x)
{
    // Sigmoid function in the range 0 to 1 spanning -2 to +2.

    half t = max(1.0 - abs(x / 2.0), 0.0);
    half y = 1.0 + sign(x) * (1.0 - t * t);

    return y / 2.0;
}

half glow_fwd(half ycIn, half glowGainIn, half glowMid)
{
    half glowGainOut;

    if (ycIn <= 2.0 / 3.0 * glowMid)
        glowGainOut = glowGainIn;
    else if (ycIn >= 2.0 * glowMid)
        glowGainOut = 0.0;
    else
        glowGainOut = glowGainIn * (glowMid / ycIn - 1.0 / 2.0);

    return glowGainOut;
}

/*
half cubic_basis_shaper
(
    half x,
    half w   // full base width of the shaper function (in degrees)
)
{
    half M[4][4] = {
        { -1.0 / 6,  3.0 / 6, -3.0 / 6,  1.0 / 6 },
        {  3.0 / 6, -6.0 / 6,  3.0 / 6,  0.0 / 6 },
        { -3.0 / 6,  0.0 / 6,  3.0 / 6,  0.0 / 6 },
        {  1.0 / 6,  4.0 / 6,  1.0 / 6,  0.0 / 6 }
    };

    half knots[5] = {
        -w / 2.0,
        -w / 4.0,
             0.0,
         w / 4.0,
         w / 2.0
    };

    half y = 0.0;
    if ((x > knots[0]) && (x < knots[4]))
    {
        half knot_coord = (x - knots[0]) * 4.0 / w;
        int j = knot_coord;
        half t = knot_coord - j;

        half monomials[4] = { t*t*t, t*t, t, 1.0 };

        // (if/else structure required for compatibility with CTL < v1.5.)
        if (j == 3)
        {
            y = monomials[0] * M[0][0] + monomials[1] * M[1][0] +
                monomials[2] * M[2][0] + monomials[3] * M[3][0];
        }
        else if (j == 2)
        {
            y = monomials[0] * M[0][1] + monomials[1] * M[1][1] +
                monomials[2] * M[2][1] + monomials[3] * M[3][1];
        }
        else if (j == 1)
        {
            y = monomials[0] * M[0][2] + monomials[1] * M[1][2] +
                monomials[2] * M[2][2] + monomials[3] * M[3][2];
        }
        else if (j == 0)
        {
            y = monomials[0] * M[0][3] + monomials[1] * M[1][3] +
                monomials[2] * M[2][3] + monomials[3] * M[3][3];
        }
        else
        {
            y = 0.0;
        }
    }

    return y * 3.0 / 2.0;
}
*/

static const half3x3 M = {
     0.5, -1.0, 0.5,
    -1.0,  1.0, 0.0,
     0.5,  0.5, 0.0
};

half segmented_spline_c5_fwd(half x)
{
    const half coefsLow[6] = { -4.0000000000, -4.0000000000, -3.1573765773, -0.4852499958, 1.8477324706, 1.8477324706 }; // coefs for B-spline between minPoint and midPoint (units of log luminance)
    const half coefsHigh[6] = { -0.7185482425, 2.0810307172, 3.6681241237, 4.0000000000, 4.0000000000, 4.0000000000 }; // coefs for B-spline between midPoint and maxPoint (units of log luminance)
    const half2 minPoint = half2(0.18 * exp2(-15.0), 0.0001); // {luminance, luminance} linear extension below this
    const half2 midPoint = half2(0.18, 0.48); // {luminance, luminance}
    const half2 maxPoint = half2(0.18 * exp2(18.0), 10000.0); // {luminance, luminance} linear extension above this
    const half slopeLow = 0.0; // log-log slope of low linear extension
    const half slopeHigh = 0.0; // log-log slope of high linear extension

    const int N_KNOTS_LOW = 4;
    const int N_KNOTS_HIGH = 4;

    // Check for negatives or zero before taking the log. If negative or zero,
    // set to ACESMIN.1
    float xCheck = x;
    if (xCheck <= 0.0) xCheck = 0.00006103515; // = pow(2.0, -14.0);

    half logx = log10(xCheck);
    half logy;

    if (logx <= log10(minPoint.x))
    {
        logy = logx * slopeLow + (log10(minPoint.y) - slopeLow * log10(minPoint.x));
    }
    else if ((logx > log10(minPoint.x)) && (logx < log10(midPoint.x)))
    {
        half knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(minPoint.x)) / (log10(midPoint.x) - log10(minPoint.x));
        int j = knot_coord;
        half t = knot_coord - j;

        half3 cf = half3(coefsLow[j], coefsLow[j + 1], coefsLow[j + 2]);
        half3 monomials = half3(t * t, t, 1.0);
        logy = dot(monomials, mul(M, cf));
    }
    else if ((logx >= log10(midPoint.x)) && (logx < log10(maxPoint.x)))
    {
        half knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(midPoint.x)) / (log10(maxPoint.x) - log10(midPoint.x));
        int j = knot_coord;
        half t = knot_coord - j;

        half3 cf = half3(coefsHigh[j], coefsHigh[j + 1], coefsHigh[j + 2]);
        half3 monomials = half3(t * t, t, 1.0);
        logy = dot(monomials, mul(M, cf));
    }
    else
    { //if (logIn >= log10(maxPoint.x)) {
        logy = logx * slopeHigh + (log10(maxPoint.y) - slopeHigh * log10(maxPoint.x));
    }

    return pow(10.0, logy);
}

half segmented_spline_c9_fwd(half x)
{
    const half coefsLow[10] = { -1.6989700043, -1.6989700043, -1.4779000000, -1.2291000000, -0.8648000000, -0.4480000000, 0.0051800000, 0.4511080334, 0.9113744414, 0.9113744414 }; // coefs for B-spline between minPoint and midPoint (units of log luminance)
    const half coefsHigh[10] = { 0.5154386965, 0.8470437783, 1.1358000000, 1.3802000000, 1.5197000000, 1.5985000000, 1.6467000000, 1.6746091357, 1.6878733390, 1.6878733390 }; // coefs for B-spline between midPoint and maxPoint (units of log luminance)
    const half2 minPoint = half2(segmented_spline_c5_fwd(0.18 * exp2(-6.5)), 0.02); // {luminance, luminance} linear extension below this
    const half2 midPoint = half2(segmented_spline_c5_fwd(0.18), 4.8); // {luminance, luminance}
    const half2 maxPoint = half2(segmented_spline_c5_fwd(0.18 * exp2(6.5)), 48.0); // {luminance, luminance} linear extension above this
    const half slopeLow = 0.0; // log-log slope of low linear extension
    const half slopeHigh = 0.04; // log-log slope of high linear extension

    const int N_KNOTS_LOW = 8;
    const int N_KNOTS_HIGH = 8;

    // Check for negatives or zero before taking the log. If negative or zero,
    // set to OCESMIN.
    half xCheck = x;
    if (xCheck <= 0.0) xCheck = 1e-4;

    half logx = log10(xCheck);
    half logy;

    if (logx <= log10(minPoint.x))
    {
        logy = logx * slopeLow + (log10(minPoint.y) - slopeLow * log10(minPoint.x));
    }
    else if ((logx > log10(minPoint.x)) && (logx < log10(midPoint.x)))
    {
        half knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(minPoint.x)) / (log10(midPoint.x) - log10(minPoint.x));
        int j = knot_coord;
        half t = knot_coord - j;

        half3 cf = half3(coefsLow[j], coefsLow[j + 1], coefsLow[j + 2]);
        half3 monomials = half3(t * t, t, 1.0);
        logy = dot(monomials, mul(M, cf));
    }
    else if ((logx >= log10(midPoint.x)) && (logx < log10(maxPoint.x)))
    {
        half knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(midPoint.x)) / (log10(maxPoint.x) - log10(midPoint.x));
        int j = knot_coord;
        half t = knot_coord - j;

        half3 cf = half3(coefsHigh[j], coefsHigh[j + 1], coefsHigh[j + 2]);
        half3 monomials = half3(t * t, t, 1.0);
        logy = dot(monomials, mul(M, cf));
    }
    else
    { //if (logIn >= log10(maxPoint.x)) {
        logy = logx * slopeHigh + (log10(maxPoint.y) - slopeHigh * log10(maxPoint.x));
    }

    return pow(10.0, logy);
}

static const half RRT_GLOW_GAIN = 0.05;
static const half RRT_GLOW_MID = 0.08;

static const half RRT_RED_SCALE = 0.82;
static const half RRT_RED_PIVOT = 0.03;
static const half RRT_RED_HUE = 0.0;
static const half RRT_RED_WIDTH = 135.0;

static const half RRT_SAT_FACTOR = 0.96;

half3 RRT(half3 aces)
{
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
        hueWeight = smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centeredHue / RRT_RED_WIDTH));
        hueWeight *= hueWeight;
    }

    aces.r += hueWeight * saturation * (RRT_RED_PIVOT - aces.r) * (1.0 - RRT_RED_SCALE);

    // --- ACES to RGB rendering space --- //
    aces = clamp(aces, 0.0, HALF_MAX);  // avoids saturated negative colors from becoming positive in the matrix
    half3 rgbPre = mul(AP0_2_AP1_MAT, aces);
    rgbPre = clamp(rgbPre, 0, HALF_MAX);

    // --- Global desaturation --- //
    //rgbPre = mul(RRT_SAT_MAT, rgbPre);
    rgbPre = lerp(dot(rgbPre, AP1_RGB2Y).xxx, rgbPre, RRT_SAT_FACTOR.xxx);

    // --- Apply the tonescale independently in rendering-space RGB --- //
    half3 rgbPost;
    rgbPost.x = segmented_spline_c5_fwd(rgbPre.x);
    rgbPost.y = segmented_spline_c5_fwd(rgbPre.y);
    rgbPost.z = segmented_spline_c5_fwd(rgbPre.z);

    // --- RGB rendering space to OCES --- //
    half3 rgbOces = mul(AP1_2_AP0_MAT, rgbPost);

    return rgbOces;
}

//
// Output Device Transform
//
half3 Y_2_linCV(half3 Y, half Ymax, half Ymin)
{
    return (Y - Ymin) / (Ymax - Ymin);
}

half3 XYZ_2_xyY(half3 XYZ)
{
    half divisor = max(dot(XYZ, (1.0).xxx), 1e-4);
    return half3(XYZ.xy / divisor, XYZ.y);
}

half3 xyY_2_XYZ(half3 xyY)
{
    half m = xyY.z / max(xyY.y, 1e-4);
    half3 XYZ = half3(xyY.xz, (1.0 - xyY.x - xyY.y));
    XYZ.xz *= m;
    return XYZ;
}

static const half DIM_SURROUND_GAMMA = 0.9811;

half3 darkSurround_to_dimSurround(half3 linearCV)
{
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    half3 xyY = XYZ_2_xyY(XYZ);
    xyY.z = clamp(xyY.z, 0.0, HALF_MAX);
    xyY.z = pow(xyY.z, DIM_SURROUND_GAMMA);
    XYZ = xyY_2_XYZ(xyY);

    return mul(XYZ_2_AP1_MAT, XYZ);
}

half moncurve_r(half y, half gamma, half offs)
{
    // Reverse monitor curve
    half x;
    const half yb = pow(offs * gamma / ((gamma - 1.0) * (1.0 + offs)), gamma);
    const half rs = pow((gamma - 1.0) / offs, gamma - 1.0) * pow((1.0 + offs) / gamma, gamma);
    if (y >= yb)
        x = (1.0 + offs) * pow(y, 1.0 / gamma) - offs;
    else
        x = y * rs;
    return x;
}

half bt1886_r(half L, half gamma, half Lw, half Lb)
{
    // The reference EOTF specified in Rec. ITU-R BT.1886
    // L = a(max[(V+b),0])^g
    half a = pow(pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma), gamma);
    half b = pow(Lb, 1.0 / gamma) / (pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma));
    half V = pow(max(L / a, 0.0), 1.0 / gamma) - b;
    return V;
}

half roll_white_fwd(
    half x,       // color value to adjust (white scaled to around 1.0)
    half new_wht, // white adjustment (e.g. 0.9 for 10% darkening)
    half width    // adjusted width (e.g. 0.25 for top quarter of the tone scale)
    )
{
    const half x0 = -1.0;
    const half x1 = x0 + width;
    const half y0 = -new_wht;
    const half y1 = x1;
    const half m1 = (x1 - x0);
    const half a = y0 - y1 + m1;
    const half b = 2.0 * (y1 - y0) - m1;
    const half c = y0;
    const half t = (-x - x0) / (x1 - x0);
    half o = 0.0;
    if (t < 0.0)
        o = -(t * b + c);
    else if (t > 1.0)
        o = x;
    else
        o = -((t * a + b) * t + c);
    return o;
}

half3 linear_to_sRGB(half3 x)
{
    return (x <= 0.0031308 ? (x * 12.9232102) : 1.055 * pow(x, 1.0 / 2.4) - 0.055);
}

half3 linear_to_bt1886(half3 x, half gamma, half Lw, half Lb)
{
    // Good enough approximation for now, may consider using the exact formula instead
    // TODO: Experiment
    return pow(max(x, 0.0), 1.0 / 2.4);

    // Correct implementation (Reference EOTF specified in Rec. ITU-R BT.1886) :
    // L = a(max[(V+b),0])^g
    half invgamma = 1.0 / gamma;
    half p_Lw = pow(Lw, invgamma);
    half p_Lb = pow(Lb, invgamma);
    half3 a = pow(p_Lw - p_Lb, gamma).xxx;
    half3 b = (p_Lb / p_Lw - p_Lb).xxx;
    half3 V = pow(max(x / a, 0.0), invgamma.xxx) - b;
    return V;
}

#if defined(CUSTOM_WHITE_POINT)
half CINEMA_WHITE;
half CINEMA_BLACK;
#else
static const half CINEMA_WHITE = 48.0;
static const half CINEMA_BLACK = CINEMA_WHITE / 2400.0;
#endif

static const half ODT_SAT_FACTOR = 0.93;

// <ACEStransformID>ODT.Academy.RGBmonitor_100nits_dim.a1.0.3</ACEStransformID>
// <ACESuserName>ACES 1.0 Output - sRGB</ACESuserName>

//
// Output Device Transform - RGB computer monitor
//

//
// Summary :
//  This transform is intended for mapping OCES onto a desktop computer monitor
//  typical of those used in motion picture visual effects production. These
//  monitors may occasionally be referred to as "sRGB" displays, however, the
//  monitor for which this transform is designed does not exactly match the
//  specifications in IEC 61966-2-1:1999.
//
//  The assumed observer adapted white is D65, and the viewing environment is
//  that of a dim surround.
//
//  The monitor specified is intended to be more typical of those found in
//  visual effects production.
//
// Device Primaries :
//  Primaries are those specified in Rec. ITU-R BT.709
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.64      0.33
//              Green:        0.3       0.6
//              Blue:         0.15      0.06
//              White:        0.3127    0.329     100 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in
//  IEC 61966-2-1:1999.
//
// Signal Range:
//    This transform outputs full range code values.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical
//   of those associated with video mastering.
//
half3 ODT_RGBmonitor_100nits_dim(half3 oces)
{
    // OCES to RGB rendering space
    half3 rgbPre = mul(AP0_2_AP1_MAT, oces);

    // Apply the tonescale independently in rendering-space RGB
    half3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd(rgbPre.x);
    rgbPost.y = segmented_spline_c9_fwd(rgbPre.y);
    rgbPost.z = segmented_spline_c9_fwd(rgbPre.z);

    // Scale luminance to linear code value
    half3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

     // Apply gamma adjustment to compensate for dim surround
    linearCV = darkSurround_to_dimSurround(linearCV);

    // Apply desaturation to compensate for luminance difference
    //linearCV = mul(ODT_SAT_MAT, linearCV);
    linearCV = lerp(dot(linearCV, AP1_RGB2Y).xxx, linearCV, ODT_SAT_FACTOR.xxx);

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    // Apply CAT from ACES white point to assumed observer adapted white point
    XYZ = mul(D60_2_D65_CAT, XYZ);

    // CIE XYZ to display primaries
    linearCV = mul(XYZ_2_REC709_MAT, XYZ);

    // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = saturate(linearCV);

    // TODO: Revisit when it is possible to deactivate Unity default framebuffer encoding
    // with sRGB opto-electrical transfer function (OETF).
    /*
    // Encode linear code values with transfer function
    half3 outputCV;
    // moncurve_r with gamma of 2.4 and offset of 0.055 matches the EOTF found in IEC 61966-2-1:1999 (sRGB)
    const half DISPGAMMA = 2.4;
    const half OFFSET = 0.055;
    outputCV.x = moncurve_r(linearCV.x, DISPGAMMA, OFFSET);
    outputCV.y = moncurve_r(linearCV.y, DISPGAMMA, OFFSET);
    outputCV.z = moncurve_r(linearCV.z, DISPGAMMA, OFFSET);

    outputCV = linear_to_sRGB(linearCV);
    */

    // Unity already draws to a sRGB target
    return linearCV;
}

// <ACEStransformID>ODT.Academy.RGBmonitor_D60sim_100nits_dim.a1.0.3</ACEStransformID>
// <ACESuserName>ACES 1.0 Output - sRGB (D60 sim.)</ACESuserName>

//
// Output Device Transform - RGB computer monitor (D60 simulation)
//

//
// Summary :
//  This transform is intended for mapping OCES onto a desktop computer monitor
//  typical of those used in motion picture visual effects production. These
//  monitors may occasionally be referred to as "sRGB" displays, however, the
//  monitor for which this transform is designed does not exactly match the
//  specifications in IEC 61966-2-1:1999.
//
//  The assumed observer adapted white is D60, and the viewing environment is
//  that of a dim surround.
//
//  The monitor specified is intended to be more typical of those found in
//  visual effects production.
//
// Device Primaries :
//  Primaries are those specified in Rec. ITU-R BT.709
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.64      0.33
//              Green:        0.3       0.6
//              Blue:         0.15      0.06
//              White:        0.3127    0.329     100 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in
//  IEC 61966-2-1:1999.
//
// Signal Range:
//    This transform outputs full range code values.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.32168      0.33767
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical
//   of those associated with video mastering.
//
half3 ODT_RGBmonitor_D60sim_100nits_dim(half3 oces)
{
    // OCES to RGB rendering space
    half3 rgbPre = mul(AP0_2_AP1_MAT, oces);

    // Apply the tonescale independently in rendering-space RGB
    half3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd(rgbPre.x);
    rgbPost.y = segmented_spline_c9_fwd(rgbPre.y);
    rgbPost.z = segmented_spline_c9_fwd(rgbPre.z);

    // Scale luminance to linear code value
    half3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

    // --- Compensate for different white point being darker  --- //
    // This adjustment is to correct an issue that exists in ODTs where the device
    // is calibrated to a white chromaticity other than D60. In order to simulate
    // D60 on such devices, unequal code values are sent to the display to achieve
    // neutrals at D60. In order to produce D60 on a device calibrated to the DCI
    // white point (i.e. equal code values yield CIE x,y chromaticities of 0.314,
    // 0.351) the red channel is higher than green and blue to compensate for the
    // "greenish" DCI white. This is the correct behavior but it means that as
    // highlight increase, the red channel will hit the device maximum first and
    // clip, resulting in a chromaticity shift as the green and blue channels
    // continue to increase.
    // To avoid this clipping error, a slight scale factor is applied to allow the
    // ODTs to simulate D60 within the D65 calibration white point.

    // Scale and clamp white to avoid casted highlights due to D60 simulation
    const half SCALE = 0.955;
    linearCV = min(linearCV, 1.0) * SCALE;

    // Apply gamma adjustment to compensate for dim surround
    linearCV = darkSurround_to_dimSurround(linearCV);

    // Apply desaturation to compensate for luminance difference
    //linearCV = mul(ODT_SAT_MAT, linearCV);
    linearCV = lerp(dot(linearCV, AP1_RGB2Y).xxx, linearCV, ODT_SAT_FACTOR.xxx);

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    // CIE XYZ to display primaries
    linearCV = mul(XYZ_2_REC709_MAT, XYZ);

    // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = saturate(linearCV);

    // TODO: Revisit when it is possible to deactivate Unity default framebuffer encoding
    // with sRGB opto-electrical transfer function (OETF).
    /*
    // Encode linear code values with transfer function
    half3 outputCV;
    // moncurve_r with gamma of 2.4 and offset of 0.055 matches the EOTF found in IEC 61966-2-1:1999 (sRGB)
    const half DISPGAMMA = 2.4;
    const half OFFSET = 0.055;
    outputCV.x = moncurve_r(linearCV.x, DISPGAMMA, OFFSET);
    outputCV.y = moncurve_r(linearCV.y, DISPGAMMA, OFFSET);
    outputCV.z = moncurve_r(linearCV.z, DISPGAMMA, OFFSET);

    outputCV = linear_to_sRGB(linearCV);
    */

    // Unity already draws to a sRGB target
    return linearCV;
}

// <ACEStransformID>ODT.Academy.Rec709_100nits_dim.a1.0.3</ACEStransformID>
// <ACESuserName>ACES 1.0 Output - Rec.709</ACESuserName>

//
// Output Device Transform - Rec709
//

//
// Summary :
//  This transform is intended for mapping OCES onto a Rec.709 broadcast monitor
//  that is calibrated to a D65 white point at 100 cd/m^2. The assumed observer
//  adapted white is D65, and the viewing environment is a dim surround.
//
//  A possible use case for this transform would be HDTV/video mastering.
//
// Device Primaries :
//  Primaries are those specified in Rec. ITU-R BT.709
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.64      0.33
//              Green:        0.3       0.6
//              Blue:         0.15      0.06
//              White:        0.3127    0.329     100 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in
//  Rec. ITU-R BT.1886.
//
// Signal Range:
//    By default, this transform outputs full range code values. If instead a
//    SMPTE "legal" signal is desired, there is a runtime flag to output
//    SMPTE legal signal. In ctlrender, this can be achieved by appending
//    '-param1 legalRange 1' after the '-ctl odt.ctl' string.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical
//   of those associated with video mastering.
//
half3 ODT_Rec709_100nits_dim(half3 oces)
{
    // OCES to RGB rendering space
    half3 rgbPre = mul(AP0_2_AP1_MAT, oces);

    // Apply the tonescale independently in rendering-space RGB
    half3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd(rgbPre.x);
    rgbPost.y = segmented_spline_c9_fwd(rgbPre.y);
    rgbPost.z = segmented_spline_c9_fwd(rgbPre.z);

    // Scale luminance to linear code value
    half3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

    // Apply gamma adjustment to compensate for dim surround
    linearCV = darkSurround_to_dimSurround(linearCV);

    // Apply desaturation to compensate for luminance difference
    //linearCV = mul(ODT_SAT_MAT, linearCV);
    linearCV = lerp(dot(linearCV, AP1_RGB2Y).xxx, linearCV, ODT_SAT_FACTOR.xxx);

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    // Apply CAT from ACES white point to assumed observer adapted white point
    XYZ = mul(D60_2_D65_CAT, XYZ);

    // CIE XYZ to display primaries
    linearCV = mul(XYZ_2_REC709_MAT, XYZ);

    // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = saturate(linearCV);

    // Encode linear code values with transfer function
    const half DISPGAMMA = 2.4;
    const half L_W = 1.0;
    const half L_B = 0.0;
    half3 outputCV = linear_to_bt1886(linearCV, DISPGAMMA, L_W, L_B);

    // TODO: Implement support for legal range.

    // NOTE: Unity framebuffer encoding is encoded with sRGB opto-electrical transfer function (OETF)
    // by default which will result in double perceptual encoding, thus for now if one want to use
    // this ODT, he needs to decode its output with sRGB electro-optical transfer function (EOTF) to
    // compensate for Unity default behaviour.

    return outputCV;
}

// <ACEStransformID>ODT.Academy.Rec709_D60sim_100nits_dim.a1.0.3</ACEStransformID>
// <ACESuserName>ACES 1.0 Output - Rec.709 (D60 sim.)</ACESuserName>

//
// Output Device Transform - Rec709 (D60 simulation)
//

//
// Summary :
//  This transform is intended for mapping OCES onto a Rec.709 broadcast monitor
//  that is calibrated to a D65 white point at 100 cd/m^2. The assumed observer
//  adapted white is D60, and the viewing environment is a dim surround.
//
//  A possible use case for this transform would be cinema "soft-proofing".
//
// Device Primaries :
//  Primaries are those specified in Rec. ITU-R BT.709
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.64      0.33
//              Green:        0.3       0.6
//              Blue:         0.15      0.06
//              White:        0.3127    0.329     100 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in
//  Rec. ITU-R BT.1886.
//
// Signal Range:
//    By default, this transform outputs full range code values. If instead a
//    SMPTE "legal" signal is desired, there is a runtime flag to output
//    SMPTE legal signal. In ctlrender, this can be achieved by appending
//    '-param1 legalRange 1' after the '-ctl odt.ctl' string.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.32168      0.33767
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical
//   of those associated with video mastering.
//
half3 ODT_Rec709_D60sim_100nits_dim(half3 oces)
{
    // OCES to RGB rendering space
    half3 rgbPre = mul(AP0_2_AP1_MAT, oces);

    // Apply the tonescale independently in rendering-space RGB
    half3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd(rgbPre.x);
    rgbPost.y = segmented_spline_c9_fwd(rgbPre.y);
    rgbPost.z = segmented_spline_c9_fwd(rgbPre.z);

    // Scale luminance to linear code value
    half3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

    // --- Compensate for different white point being darker  --- //
    // This adjustment is to correct an issue that exists in ODTs where the device
    // is calibrated to a white chromaticity other than D60. In order to simulate
    // D60 on such devices, unequal code values must be sent to the display to achieve
    // the chromaticities of D60. More specifically, in order to produce D60 on a device
    // calibrated to a D65 white point (i.e. equal code values yield CIE x,y
    // chromaticities of 0.3127, 0.329) the red channel must be slightly higher than
    // that of green and blue in order to compensate for the relatively more "blue-ish"
    // D65 white. This unequalness of color channels is the correct behavior but it
    // means that as neutral highlights increase, the red channel will hit the
    // device maximum first and clip, resulting in a small chromaticity shift as the
    // green and blue channels continue to increase to their maximums.
    // To avoid this clipping error, a slight scale factor is applied to allow the
    // ODTs to simulate D60 within the D65 calibration white point.

    // Scale and clamp white to avoid casted highlights due to D60 simulation
    const half SCALE = 0.955;
    linearCV = min(linearCV, 1.0) * SCALE;

    // Apply gamma adjustment to compensate for dim surround
    linearCV = darkSurround_to_dimSurround(linearCV);

    // Apply desaturation to compensate for luminance difference
    //linearCV = mul(ODT_SAT_MAT, linearCV);
    linearCV = lerp(dot(linearCV, AP1_RGB2Y).xxx, linearCV, ODT_SAT_FACTOR.xxx);

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    // CIE XYZ to display primaries
    linearCV = mul(XYZ_2_REC709_MAT, XYZ);

    // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = saturate(linearCV);

    // Encode linear code values with transfer function
    const half DISPGAMMA = 2.4;
    const half L_W = 1.0;
    const half L_B = 0.0;
    half3 outputCV = linear_to_bt1886(linearCV, DISPGAMMA, L_W, L_B);

    // TODO: Implement support for legal range.

    // NOTE: Unity framebuffer encoding is encoded with sRGB opto-electrical transfer function (OETF)
    // by default which will result in double perceptual encoding, thus for now if one want to use
    // this ODT, he needs to decode its output with sRGB electro-optical transfer function (EOTF) to
    // compensate for Unity default behaviour.

    return outputCV;
}

// <ACEStransformID>ODT.Academy.Rec2020_100nits_dim.a1.0.3</ACEStransformID>
// <ACESuserName>ACES 1.0 Output - Rec.2020</ACESuserName>

//
// Output Device Transform - Rec2020
//

//
// Summary :
//  This transform is intended for mapping OCES onto a Rec.2020 broadcast
//  monitor that is calibrated to a D65 white point at 100 cd/m^2. The assumed
//  observer adapted white is D65, and the viewing environment is that of a dim
//  surround.
//
//  A possible use case for this transform would be UHDTV/video mastering.
//
// Device Primaries :
//  Primaries are those specified in Rec. ITU-R BT.2020
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.708     0.292
//              Green:        0.17      0.797
//              Blue:         0.131     0.046
//              White:        0.3127    0.329     100 cd/m^2
//
// Display EOTF :
//  The reference electro-optical transfer function specified in
//  Rec. ITU-R BT.1886.
//
// Signal Range:
//    By default, this transform outputs full range code values. If instead a
//    SMPTE "legal" signal is desired, there is a runtime flag to output
//    SMPTE legal signal. In ctlrender, this can be achieved by appending
//    '-param1 legalRange 1' after the '-ctl odt.ctl' string.
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.3127       0.329
//
// Viewing Environment:
//   This ODT has a compensation for viewing environment variables more typical
//   of those associated with video mastering.
//

half3 ODT_Rec2020_100nits_dim(half3 oces)
{
    // OCES to RGB rendering space
    half3 rgbPre = mul(AP0_2_AP1_MAT, oces);

    // Apply the tonescale independently in rendering-space RGB
    half3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd(rgbPre.x);
    rgbPost.y = segmented_spline_c9_fwd(rgbPre.y);
    rgbPost.z = segmented_spline_c9_fwd(rgbPre.z);

    // Scale luminance to linear code value
    half3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

    // Apply gamma adjustment to compensate for dim surround
    linearCV = darkSurround_to_dimSurround(linearCV);

    // Apply desaturation to compensate for luminance difference
    //linearCV = mul(ODT_SAT_MAT, linearCV);
    linearCV = lerp(dot(linearCV, AP1_RGB2Y).xxx, linearCV, ODT_SAT_FACTOR.xxx);

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    // Apply CAT from ACES white point to assumed observer adapted white point
    XYZ = mul(D60_2_D65_CAT, XYZ);

    // CIE XYZ to display primaries
    linearCV = mul(XYZ_2_REC2020_MAT, XYZ);

    // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = saturate(linearCV);

    // Encode linear code values with transfer function
    const half DISPGAMMA = 2.4;
    const half L_W = 1.0;
    const half L_B = 0.0;
    half3 outputCV = linear_to_bt1886(linearCV, DISPGAMMA, L_W, L_B);

    // TODO: Implement support for legal range.

    // NOTE: Unity framebuffer encoding is encoded with sRGB opto-electrical transfer function (OETF)
    // by default which will result in double perceptual encoding, thus for now if one want to use
    // this ODT, he needs to decode its output with sRGB electro-optical transfer function (EOTF) to
    // compensate for Unity default behaviour.

    return outputCV;
}

// <ACEStransformID>ODT.Academy.P3DCI_48nits.a1.0.3</ACEStransformID>
// <ACESuserName>ACES 1.0 Output - P3-DCI</ACESuserName>

//
// Output Device Transform - P3DCI (D60 Simulation)
//

//
// Summary :
//  This transform is intended for mapping OCES onto a P3 digital cinema
//  projector that is calibrated to a DCI white point at 48 cd/m^2. The assumed
//  observer adapted white is D60, and the viewing environment is that of a dark
//  theater.
//
// Device Primaries :
//  CIE 1931 chromaticities:  x         y         Y
//              Red:          0.68      0.32
//              Green:        0.265     0.69
//              Blue:         0.15      0.06
//              White:        0.314     0.351     48 cd/m^2
//
// Display EOTF :
//  Gamma: 2.6
//
// Assumed observer adapted white point:
//         CIE 1931 chromaticities:    x            y
//                                     0.32168      0.33767
//
// Viewing Environment:
//  Environment specified in SMPTE RP 431-2-2007
//
half3 ODT_P3DCI_48nits(half3 oces)
{
    // OCES to RGB rendering space
    half3 rgbPre = mul(AP0_2_AP1_MAT, oces);

    // Apply the tonescale independently in rendering-space RGB
    half3 rgbPost;
    rgbPost.x = segmented_spline_c9_fwd(rgbPre.x);
    rgbPost.y = segmented_spline_c9_fwd(rgbPre.y);
    rgbPost.z = segmented_spline_c9_fwd(rgbPre.z);

    // Scale luminance to linear code value
    half3 linearCV = Y_2_linCV(rgbPost, CINEMA_WHITE, CINEMA_BLACK);

    // --- Compensate for different white point being darker  --- //
    // This adjustment is to correct an issue that exists in ODTs where the device
    // is calibrated to a white chromaticity other than D60. In order to simulate
    // D60 on such devices, unequal code values are sent to the display to achieve
    // neutrals at D60. In order to produce D60 on a device calibrated to the DCI
    // white point (i.e. equal code values yield CIE x,y chromaticities of 0.314,
    // 0.351) the red channel is higher than green and blue to compensate for the
    // "greenish" DCI white. This is the correct behavior but it means that as
    // highlight increase, the red channel will hit the device maximum first and
    // clip, resulting in a chromaticity shift as the green and blue channels
    // continue to increase.
    // To avoid this clipping error, a slight scale factor is applied to allow the
    // ODTs to simulate D60 within the D65 calibration white point. However, the
    // magnitude of the scale factor required for the P3DCI ODT was considered too
    // large. Therefore, the scale factor was reduced and the additional required
    // compression was achieved via a reshaping of the highlight rolloff in
    // conjunction with the scale. The shape of this rolloff was determined
    // throught subjective experiments and deemed to best reproduce the
    // "character" of the highlights in the P3D60 ODT.

    // Roll off highlights to avoid need for as much scaling
    const half NEW_WHT = 0.918;
    const half ROLL_WIDTH = 0.5;
    linearCV.x = roll_white_fwd(linearCV.x, NEW_WHT, ROLL_WIDTH);
    linearCV.y = roll_white_fwd(linearCV.y, NEW_WHT, ROLL_WIDTH);
    linearCV.z = roll_white_fwd(linearCV.z, NEW_WHT, ROLL_WIDTH);

    // Scale and clamp white to avoid casted highlights due to D60 simulation
    const half SCALE = 0.96;
    linearCV = min(linearCV, NEW_WHT) * SCALE;

    // Convert to display primary encoding
    // Rendering space RGB to XYZ
    half3 XYZ = mul(AP1_2_XYZ_MAT, linearCV);

    // CIE XYZ to display primaries
    linearCV = mul(XYZ_2_DCIP3_MAT, XYZ);

    // Handle out-of-gamut values
    // Clip values < 0 or > 1 (i.e. projecting outside the display primaries)
    linearCV = saturate(linearCV);

    // Encode linear code values with transfer function
    const half DISPGAMMA = 2.6;
    half3 outputCV = pow(linearCV, 1.0 / DISPGAMMA);

    // NOTE: Unity framebuffer encoding is encoded with sRGB opto-electrical transfer function (OETF)
    // by default which will result in double perceptual encoding, thus for now if one want to use
    // this ODT, he needs to decode its output with sRGB electro-optical transfer function (EOTF) to
    // compensate for Unity default behaviour.

    return outputCV;
}

#endif // __ACES__
