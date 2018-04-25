#ifndef __DEPTH_OF_FIELD__
#define __DEPTH_OF_FIELD__

#if SHADER_TARGET >= 50
    // Use separate texture/sampler objects on Shader Model 5.0
    #define SEPARATE_TEXTURE_SAMPLER
    #define DOF_DECL_TEX2D(tex) Texture2D tex; SamplerState sampler##tex
    #define DOF_TEX2D(tex, coord) tex.Sample(sampler##tex, coord)
#else
    #define DOF_DECL_TEX2D(tex) sampler2D tex
    #define DOF_TEX2D(tex, coord) tex2D(tex, coord)
#endif

#include "Common.cginc"
#include "DiskKernels.cginc"

DOF_DECL_TEX2D(_CameraDepthTexture);
DOF_DECL_TEX2D(_CameraMotionVectorsTexture);
DOF_DECL_TEX2D(_CoCTex);

// Camera parameters
float _Distance;
float _LensCoeff;  // f^2 / (N * (S1 - f) * film_width * 2)
float _MaxCoC;
float _RcpMaxCoC;
float _RcpAspect;
half3 _TaaParams; // Jitter.x, Jitter.y, Blending

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

// CoC calculation
half4 FragCoC(VaryingsDOF i) : SV_Target
{
    float depth = LinearEyeDepth(DOF_TEX2D(_CameraDepthTexture, i.uv));
    half coc = (depth - _Distance) * _LensCoeff / max(depth, 1e-5);
    return saturate(coc * 0.5 * _RcpMaxCoC + 0.5);
}

// Temporal filter
half4 FragTempFilter(VaryingsDOF i) : SV_Target
{
    float3 uvOffs = _MainTex_TexelSize.xyy * float3(1, 1, 0);

#if defined(SEPARATE_TEXTURE_SAMPLER)

    half4 cocTL = _CoCTex.GatherRed(sampler_CoCTex, i.uv - uvOffs.xy * 0.5); // top-left
    half4 cocBR = _CoCTex.GatherRed(sampler_CoCTex, i.uv + uvOffs.xy * 0.5); // bottom-right
    half coc1 = cocTL.x; // top
    half coc2 = cocTL.z; // left
    half coc3 = cocBR.x; // bottom
    half coc4 = cocBR.z; // right

#else

    half coc1 = DOF_TEX2D(_CoCTex, i.uv - uvOffs.xz).r; // top
    half coc2 = DOF_TEX2D(_CoCTex, i.uv - uvOffs.zy).r; // left
    half coc3 = DOF_TEX2D(_CoCTex, i.uv + uvOffs.zy).r; // bottom
    half coc4 = DOF_TEX2D(_CoCTex, i.uv + uvOffs.xz).r; // right

#endif

    // Dejittered center sample.
    half coc0 = DOF_TEX2D(_CoCTex, i.uv - _TaaParams.xy).r;

    // CoC dilation: determine the closest point in the four neighbors.
    float3 closest = float3(0, 0, coc0);
    closest = coc1 < closest.z ? float3(-uvOffs.xz, coc1) : closest;
    closest = coc2 < closest.z ? float3(-uvOffs.zy, coc2) : closest;
    closest = coc3 < closest.z ? float3(+uvOffs.zy, coc3) : closest;
    closest = coc4 < closest.z ? float3(+uvOffs.xz, coc4) : closest;

    // Sample the history buffer with the motion vector at the closest point.
    float2 motion = DOF_TEX2D(_CameraMotionVectorsTexture, i.uv + closest.xy).xy;
    half cocHis = DOF_TEX2D(_MainTex, i.uv - motion).r;

    // Neighborhood clamping.
    half cocMin = closest.z;
    half cocMax = max(max(max(max(coc0, coc1), coc2), coc3), coc4);
    cocHis = clamp(cocHis, cocMin, cocMax);

    // Blend with the history.
    return lerp(coc0, cocHis, _TaaParams.z);
}

// Prefilter: downsampling and premultiplying.
half4 FragPrefilter(VaryingsDOF i) : SV_Target
{
#if defined(SEPARATE_TEXTURE_SAMPLER)

    // Sample source colors.
    half4 c_r = _MainTex.GatherRed  (sampler_MainTex, i.uv);
    half4 c_g = _MainTex.GatherGreen(sampler_MainTex, i.uv);
    half4 c_b = _MainTex.GatherBlue (sampler_MainTex, i.uv);

    half3 c0 = half3(c_r.x, c_g.x, c_b.x);
    half3 c1 = half3(c_r.y, c_g.y, c_b.y);
    half3 c2 = half3(c_r.z, c_g.z, c_b.z);
    half3 c3 = half3(c_r.w, c_g.w, c_b.w);

    // Sample CoCs.
    half4 cocs = _CoCTex.Gather(sampler_CoCTex, i.uvAlt) * 2.0 - 1.0;
    half coc0 = cocs.x;
    half coc1 = cocs.y;
    half coc2 = cocs.z;
    half coc3 = cocs.w;

#else

    float3 duv = _MainTex_TexelSize.xyx * float3(0.5, 0.5, -0.5);

    // Sample source colors.
    half3 c0 = DOF_TEX2D(_MainTex, i.uv - duv.xy).rgb;
    half3 c1 = DOF_TEX2D(_MainTex, i.uv - duv.zy).rgb;
    half3 c2 = DOF_TEX2D(_MainTex, i.uv + duv.zy).rgb;
    half3 c3 = DOF_TEX2D(_MainTex, i.uv + duv.xy).rgb;

    // Sample CoCs.
    half coc0 = DOF_TEX2D(_CoCTex, i.uvAlt - duv.xy).r * 2.0 - 1.0;
    half coc1 = DOF_TEX2D(_CoCTex, i.uvAlt - duv.zy).r * 2.0 - 1.0;
    half coc2 = DOF_TEX2D(_CoCTex, i.uvAlt + duv.zy).r * 2.0 - 1.0;
    half coc3 = DOF_TEX2D(_CoCTex, i.uvAlt + duv.xy).r * 2.0 - 1.0;

#endif

    // Apply CoC and luma weights to reduce bleeding and flickering.
    float w0 = abs(coc0) / (Max3(c0) + 1.0);
    float w1 = abs(coc1) / (Max3(c1) + 1.0);
    float w2 = abs(coc2) / (Max3(c2) + 1.0);
    float w3 = abs(coc3) / (Max3(c3) + 1.0);

    // Weighted average of the color samples
    half3 avg = c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3;
    avg /= max(w0 + w1 + w2 + w3, 1e-5);

    // Select the largest CoC value.
    half coc_min = Min4(coc0, coc1, coc2, coc3);
    half coc_max = Max4(coc0, coc1, coc2, coc3);
    half coc = (-coc_min > coc_max ? coc_min : coc_max) * _MaxCoC;

    // Premultiply CoC again.
    avg *= smoothstep(0, _MainTex_TexelSize.y * 2, abs(coc));

#if defined(UNITY_COLORSPACE_GAMMA)
    avg = GammaToLinearSpace(avg);
#endif

    return half4(avg, coc);
}

// Bokeh filter with disk-shaped kernels
half4 FragBlur(VaryingsDOF i) : SV_Target
{
    half4 samp0 = DOF_TEX2D(_MainTex, i.uv);

    half4 bgAcc = 0.0; // Background: far field bokeh
    half4 fgAcc = 0.0; // Foreground: near field bokeh

    UNITY_LOOP for (int si = 0; si < kSampleCount; si++)
    {
        float2 disp = kDiskKernel[si] * _MaxCoC;
        float dist = length(disp);

        float2 duv = float2(disp.x * _RcpAspect, disp.y);
        half4 samp = DOF_TEX2D(_MainTex, i.uv + duv);

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
    half alpha = saturate(fgAcc.a);
    half3 rgb = lerp(bgAcc.rgb, fgAcc.rgb, alpha);

    return half4(rgb, alpha);
}

// Postfilter blur
half4 FragPostBlur(VaryingsDOF i) : SV_Target
{
    // 9 tap tent filter with 4 bilinear samples
    const float4 duv = _MainTex_TexelSize.xyxy * float4(0.5, 0.5, -0.5, 0);
    half4 acc;
    acc  = DOF_TEX2D(_MainTex, i.uv - duv.xy);
    acc += DOF_TEX2D(_MainTex, i.uv - duv.zy);
    acc += DOF_TEX2D(_MainTex, i.uv + duv.zy);
    acc += DOF_TEX2D(_MainTex, i.uv + duv.xy);
    return acc / 4.0;
}

#endif // __DEPTH_OF_FIELD__
