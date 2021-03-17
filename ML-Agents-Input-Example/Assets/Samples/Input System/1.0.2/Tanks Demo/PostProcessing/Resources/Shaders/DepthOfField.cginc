#ifndef __DEPTH_OF_FIELD__
#define __DEPTH_OF_FIELD__

#include "UnityCG.cginc"
#include "Common.cginc"
#include "DiskKernels.cginc"

#define PREFILTER_LUMA_WEIGHT 1

sampler2D_float _CameraDepthTexture;
sampler2D_float _HistoryCoC;
float _HistoryWeight;

// Camera parameters
float _Distance;
float _LensCoeff;  // f^2 / (N * (S1 - f) * film_width * 2)
float _MaxCoC;
float _RcpMaxCoC;
float _RcpAspect;

struct VaryingsDOF
{
    float4 pos : SV_POSITION;
    half2 uv : TEXCOORD0;
    half2 uvAlt : TEXCOORD1;
};

// Common vertex shader with single pass stereo rendering support
VaryingsDOF VertDOF(AttributesDefault v)
{
    half2 uvAlt = v.texcoord;
#if UNITY_UV_STARTS_AT_TOP
    if (_MainTex_TexelSize.y < 0.0) uvAlt.y = 1.0 - uvAlt.y;
#endif

    VaryingsDOF o;
    o.pos = UnityObjectToClipPos(v.vertex);

#if defined(UNITY_SINGLE_PASS_STEREO)
    o.uv = UnityStereoScreenSpaceUVAdjust(v.texcoord, _MainTex_ST);
    o.uvAlt = UnityStereoScreenSpaceUVAdjust(uvAlt, _MainTex_ST);
#else
    o.uv = v.texcoord;
    o.uvAlt = uvAlt;
#endif

    return o;
}

// Prefilter: CoC calculation, downsampling and premultiplying.

#if defined(PREFILTER_TAA)

// TAA enabled: use MRT to update the history buffer in the same pass.
struct PrefilterOutput
{
    half4 base : SV_Target0;
    half4 history : SV_Target1;
};
#define PrefilterSemantics

#else

// No TAA
#define PrefilterOutput half4
#define PrefilterSemantics :SV_Target

#endif

PrefilterOutput FragPrefilter(VaryingsDOF i) PrefilterSemantics
{
    float3 duv = _MainTex_TexelSize.xyx * float3(0.5, 0.5, -0.5);

    // Sample source colors.
    half3 c0 = tex2D(_MainTex, i.uv - duv.xy).rgb;
    half3 c1 = tex2D(_MainTex, i.uv - duv.zy).rgb;
    half3 c2 = tex2D(_MainTex, i.uv + duv.zy).rgb;
    half3 c3 = tex2D(_MainTex, i.uv + duv.xy).rgb;

    // Sample linear depths.
    float d0 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt - duv.xy));
    float d1 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt - duv.zy));
    float d2 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt + duv.zy));
    float d3 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uvAlt + duv.xy));
    float4 depths = float4(d0, d1, d2, d3);

    // Calculate the radiuses of CoCs at these sample points.
    float4 cocs = (depths - _Distance) * _LensCoeff / depths;
    cocs = clamp(cocs, -_MaxCoC, _MaxCoC);

#if defined(PREFILTER_TAA)
    // Get the average with the history to avoid temporal aliasing.
    half hcoc = tex2D(_HistoryCoC, i.uv).r;
    cocs = lerp(cocs, hcoc, _HistoryWeight);
#endif

    // Premultiply CoC to reduce background bleeding.
    float4 weights = saturate(abs(cocs) * _RcpMaxCoC);

#if defined(PREFILTER_LUMA_WEIGHT)
    // Apply luma weights to reduce flickering.
    // References:
    //   http://gpuopen.com/optimized-reversible-tonemapper-for-resolve/
    //   http://graphicrants.blogspot.fr/2013/12/tone-mapping.html
    weights.x *= 1.0 / (Max3(c0) + 1.0);
    weights.y *= 1.0 / (Max3(c1) + 1.0);
    weights.z *= 1.0 / (Max3(c2) + 1.0);
    weights.w *= 1.0 / (Max3(c3) + 1.0);
#endif

    // Weighted average of the color samples
    half3 avg = c0 * weights.x + c1 * weights.y + c2 * weights.z + c3 * weights.w;
    avg /= dot(weights, 1.0);

    // Output CoC = average of CoCs
    half cocmin = Min4(cocs);
    half cocmax = Max4(cocs);
    half coc = -cocmin > cocmax ? cocmin : cocmax;

    // Premultiply CoC again.
    avg *= smoothstep(0, _MainTex_TexelSize.y * 2, abs(coc));

#if defined(UNITY_COLORSPACE_GAMMA)
    avg = GammaToLinearSpace(avg);
#endif

#if defined(PREFILTER_TAA)
    PrefilterOutput output;
    output.base = half4(avg, coc);
    output.history = coc.xxxx;
    return output;
#else
    return half4(avg, coc);
#endif
}

// Bokeh filter with disk-shaped kernels
half4 FragBlur(VaryingsDOF i) : SV_Target
{
    half4 samp0 = tex2D(_MainTex, i.uv);

    half4 bgAcc = 0.0; // Background: far field bokeh
    half4 fgAcc = 0.0; // Foreground: near field bokeh

    UNITY_LOOP for (int si = 0; si < kSampleCount; si++)
    {
        float2 disp = kDiskKernel[si] * _MaxCoC;
        float dist = length(disp);

        float2 duv = float2(disp.x * _RcpAspect, disp.y);
        half4 samp = tex2D(_MainTex, i.uv + duv);

        // BG: Compare CoC of the current sample and the center sample
        // and select smaller one.
        half bgCoC = max(min(samp0.a, samp.a), 0.0);

        // Compare the CoC to the sample distance.
        // Add a small margin to smooth out.
        const half margin = _MainTex_TexelSize.y * 2;
        half bgWeight = saturate((bgCoC   - dist + margin) / margin);
        half fgWeight = saturate((-samp.a - dist + margin) / margin);

        // Cut influence from focused areas because they're darkened by CoC
        // premultiplying. This is only needed for near field.
        fgWeight *= step(_MainTex_TexelSize.y, -samp.a);

        // Accumulation
        bgAcc += half4(samp.rgb, 1.0) * bgWeight;
        fgAcc += half4(samp.rgb, 1.0) * fgWeight;
    }

    // Get the weighted average.
    bgAcc.rgb /= bgAcc.a + (bgAcc.a == 0.0); // zero-div guard
    fgAcc.rgb /= fgAcc.a + (fgAcc.a == 0.0);

    // BG: Calculate the alpha value only based on the center CoC.
    // This is a rather aggressive approximation but provides stable results.
    bgAcc.a = smoothstep(_MainTex_TexelSize.y, _MainTex_TexelSize.y * 2.0, samp0.a);

    // FG: Normalize the total of the weights.
    fgAcc.a *= UNITY_PI / kSampleCount;

    // Alpha premultiplying
    half3 rgb = 0.0;
    rgb = lerp(rgb, bgAcc.rgb, saturate(bgAcc.a));
    rgb = lerp(rgb, fgAcc.rgb, saturate(fgAcc.a));

    // Combined alpha value
    half alpha = (1.0 - saturate(bgAcc.a)) * (1.0 - saturate(fgAcc.a));

    return half4(rgb, alpha);
}

// Postfilter blur
half4 FragPostBlur(VaryingsDOF i) : SV_Target
{
    // 9-tap tent filter
    float4 duv = _MainTex_TexelSize.xyxy * float4(1, 1, -1, 0);

    half4 c0 = tex2D(_MainTex, i.uv - duv.xy);
    half4 c1 = tex2D(_MainTex, i.uv - duv.wy);
    half4 c2 = tex2D(_MainTex, i.uv - duv.zy);

    half4 c3 = tex2D(_MainTex, i.uv + duv.zw);
    half4 c4 = tex2D(_MainTex, i.uv         );
    half4 c5 = tex2D(_MainTex, i.uv + duv.xw);

    half4 c6 = tex2D(_MainTex, i.uv + duv.zy);
    half4 c7 = tex2D(_MainTex, i.uv + duv.wy);
    half4 c8 = tex2D(_MainTex, i.uv + duv.xy);

    half4 acc = c0 * 1 + c1 * 2 + c2 * 1 +
                c3 * 2 + c4 * 4 + c5 * 2 +
                c6 * 1 + c7 * 2 + c8 * 1;

    half aa =
        c0.a * c0.a * 1 + c1.a * c1.a * 2 + c2.a * c2.a * 1 +
        c3.a * c3.a * 2 + c4.a * c4.a * 4 + c5.a * c5.a * 2 +
        c6.a * c6.a * 1 + c7.a * c7.a * 2 + c8.a * c8.a * 1;

    half wb = 1.2;
    half a = (wb * acc.a - aa) / (wb * 16 - acc.a);

    acc /= 16;

    half3 rgb = acc.rgb * (1 + saturate(acc.a - a));
    return half4(rgb, a);
}

#endif // __DEPTH_OF_FIELD__
